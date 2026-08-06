//! HF Hub integration: parquet tree discovery, streaming parquet reads,
//! tokenizer download.
//!
//! Parquet streaming uses a custom `AsyncFileReader` over `reqwest` rather
//! than `object_store`'s HTTP backend, because HF's auto-converted parquet
//! URLs embed `refs%2Fconvert%2Fparquet` as a single URL path segment;
//! `ObjectStore::Path` either splits on the decoded slashes (breaking the
//! URL) or double-encodes the percent (also breaking it). Going direct with
//! reqwest keeps the URL byte-identical to what the HF API returns.

use std::collections::BTreeMap;
use std::ops::Range;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use bytes::Bytes;
use futures::FutureExt;
use futures::future::BoxFuture;
use parquet::arrow::ProjectionMask;
use parquet::arrow::async_reader::{
    AsyncFileReader, ParquetRecordBatchStream, ParquetRecordBatchStreamBuilder,
};
use parquet::errors::{ParquetError, Result as ParquetResult};
use parquet::file::metadata::{ParquetMetaData, ParquetMetaDataReader};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;
use tracing::info;

use crate::config::{CACHE_ROOT, HF_DATASET, TOKENIZER_MODEL};

const HF_API_BASE: &str = "https://huggingface.co/api/datasets";
const HF_MODEL_BASE: &str = "https://huggingface.co";
const DATASETS_SERVER_BASE: &str = "https://datasets-server.huggingface.co";

pub fn cache_dir() -> PathBuf {
    PathBuf::from(CACHE_ROOT)
}

pub fn parquet_tree_cache_path() -> PathBuf {
    cache_dir().join("sources.json")
}

pub fn tokenizer_cache_path() -> PathBuf {
    cache_dir().join("tokenizer.json")
}

// ----------------------------------------------------------------------------
// Parquet tree discovery
// ----------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParquetTree {
    /// source_name -> list of full parquet URLs for the `train` split
    pub configs: BTreeMap<String, Vec<String>>,
}

/// Fetch the HF parquet API tree. Cached on disk after first call.
pub async fn discover_parquet_tree() -> Result<ParquetTree> {
    let path = parquet_tree_cache_path();
    if path.exists() {
        let s = std::fs::read_to_string(&path)?;
        return Ok(serde_json::from_str(&s)?);
    }

    let url = format!("{HF_API_BASE}/{HF_DATASET}/parquet");
    info!("Fetching parquet tree from {url}");
    let client = build_http_client()?;
    let resp = client
        .get(&url)
        .send()
        .await
        .with_context(|| format!("GET {url}"))?
        .error_for_status()?;
    let json: serde_json::Value = resp.json().await?;

    let mut configs: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let obj = json
        .as_object()
        .ok_or_else(|| anyhow!("expected object response from parquet API"))?;
    for (config_name, splits) in obj {
        let split_obj = splits
            .as_object()
            .ok_or_else(|| anyhow!("splits not object for {config_name}"))?;
        let urls_val = split_obj
            .get("train")
            .or_else(|| split_obj.values().next())
            .ok_or_else(|| anyhow!("no splits for {config_name}"))?;
        let urls_arr = urls_val
            .as_array()
            .ok_or_else(|| anyhow!("split urls not array for {config_name}"))?;
        let urls: Vec<String> = urls_arr
            .iter()
            .map(|v| {
                v.as_str()
                    .map(String::from)
                    .ok_or_else(|| anyhow!("url not string"))
            })
            .collect::<Result<_>>()?;
        configs.insert(config_name.clone(), urls);
    }

    info!("Discovered {} configs", configs.len());
    let tree = ParquetTree { configs };
    std::fs::create_dir_all(cache_dir())?;
    crate::atomic::atomic_write(&path, serde_json::to_string_pretty(&tree)?.as_bytes())?;
    Ok(tree)
}

// ----------------------------------------------------------------------------
// Tokenizer download
// ----------------------------------------------------------------------------

pub async fn load_tokenizer() -> Result<Tokenizer> {
    let path = tokenizer_cache_path();
    if !path.exists() {
        let url = format!("{HF_MODEL_BASE}/{TOKENIZER_MODEL}/resolve/main/tokenizer.json");
        info!("Downloading tokenizer from {url}");
        let client = build_http_client()?;
        let bytes = client
            .get(&url)
            .send()
            .await?
            .error_for_status()?
            .bytes()
            .await?;
        std::fs::create_dir_all(cache_dir())?;
        std::fs::write(&path, &bytes)?;
    }
    Tokenizer::from_file(&path).map_err(|e| anyhow!("load tokenizer: {e}"))
}

// ----------------------------------------------------------------------------
// HTTP-backed AsyncFileReader for parquet streaming
// ----------------------------------------------------------------------------

pub fn build_http_client() -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .user_agent("lattice-pipeline/0.1")
        .pool_idle_timeout(std::time::Duration::from_secs(90))
        .build()
        .map_err(Into::into)
}

pub struct HttpFileReader {
    client: reqwest::Client,
    url: String,
    size: u64,
}

impl HttpFileReader {
    pub async fn open(client: reqwest::Client, url: String) -> Result<Self> {
        // Some HF endpoints respond to HEAD with no Content-Length; do a GET
        // for the first byte and read Content-Range to learn the full size.
        let resp = client
            .get(&url)
            .header(reqwest::header::RANGE, "bytes=0-0")
            .send()
            .await
            .with_context(|| format!("probe {url}"))?
            .error_for_status()?;

        let size = if let Some(cr) = resp.headers().get(reqwest::header::CONTENT_RANGE) {
            let cr = cr.to_str().context("Content-Range not ascii")?;
            cr.split('/')
                .nth(1)
                .and_then(|s| s.parse::<u64>().ok())
                .ok_or_else(|| anyhow!("malformed Content-Range: {cr}"))?
        } else if let Some(cl) = resp.headers().get(reqwest::header::CONTENT_LENGTH) {
            cl.to_str()?.parse::<u64>()?
        } else {
            return Err(anyhow!("no size info for {url}"));
        };

        Ok(Self { client, url, size })
    }

    pub fn size(&self) -> u64 {
        self.size
    }
}

impl AsyncFileReader for HttpFileReader {
    fn get_bytes(&mut self, range: Range<usize>) -> BoxFuture<'_, ParquetResult<Bytes>> {
        let url = self.url.clone();
        let client = self.client.clone();
        async move {
            let range_str = format!("bytes={}-{}", range.start, range.end - 1);
            let resp = client
                .get(&url)
                .header(reqwest::header::RANGE, range_str)
                .send()
                .await
                .map_err(|e| ParquetError::External(Box::new(e)))?
                .error_for_status()
                .map_err(|e| ParquetError::External(Box::new(e)))?;
            resp.bytes()
                .await
                .map_err(|e| ParquetError::External(Box::new(e)))
        }
        .boxed()
    }

    fn get_metadata(&mut self) -> BoxFuture<'_, ParquetResult<Arc<ParquetMetaData>>> {
        let size = self.size as usize;
        async move {
            let mut reader = ParquetMetaDataReader::new();
            reader.try_load(self, size).await?;
            Ok(Arc::new(reader.finish()?))
        }
        .boxed()
    }
}

// ----------------------------------------------------------------------------
// Public helpers
// ----------------------------------------------------------------------------

/// One call to datasets-server `/size` returns row counts for every config.
/// Much faster than reading each parquet footer.
pub async fn dataset_sizes() -> Result<BTreeMap<String, u64>> {
    let url = format!("{DATASETS_SERVER_BASE}/size?dataset={HF_DATASET}");
    info!("Fetching config sizes from {url}");
    let client = build_http_client()?;
    let resp = client
        .get(&url)
        .send()
        .await
        .with_context(|| format!("GET {url}"))?
        .error_for_status()?;
    let json: serde_json::Value = resp.json().await?;
    let configs = json
        .pointer("/size/configs")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow!("missing /size/configs in /size response"))?;
    let mut out = BTreeMap::new();
    for c in configs {
        let name = c
            .get("config")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("config entry missing 'config'"))?;
        let rows = c
            .get("num_rows")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("config entry missing 'num_rows'"))?;
        out.insert(name.to_string(), rows);
    }
    Ok(out)
}

pub async fn open_stream(
    client: reqwest::Client,
    url: &str,
    columns: &[&str],
    batch_size: usize,
) -> Result<ParquetRecordBatchStream<HttpFileReader>> {
    let reader = HttpFileReader::open(client, url.to_string()).await?;
    let builder = ParquetRecordBatchStreamBuilder::new(reader).await?;
    let schema = builder.schema().clone();
    let mut indices: Vec<usize> = Vec::with_capacity(columns.len());
    for c in columns {
        let idx = schema
            .index_of(c)
            .map_err(|_| anyhow!("column {c} not found; schema: {:?}", schema.fields()))?;
        indices.push(idx);
    }
    let mask = ProjectionMask::roots(builder.parquet_schema(), indices);
    Ok(builder
        .with_projection(mask)
        .with_batch_size(batch_size)
        .build()?)
}
