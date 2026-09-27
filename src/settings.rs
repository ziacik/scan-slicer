use anyhow::{Context, Result};
use futures_lite::future;
use oo7::{Keyring, Secret};

const ATTRIBUTES: [(&str, &str); 2] = [
    ("application", "com.github.ziacik.ScanSlicer"),
    ("type", "openai-api-key"),
];

pub fn load_openai_api_key() -> Result<Option<String>> {
    future::block_on(async {
        let keyring = open_keyring().await?;
        let items = keyring
            .search_items(&ATTRIBUTES)
            .await
            .context("could not search the system keyring")?;

        let Some(item) = items.first() else {
            return Ok(None);
        };

        let secret = item
            .secret()
            .await
            .context("could not read the OpenAI API key")?;
        let api_key = std::str::from_utf8(secret.as_bytes())
            .context("stored OpenAI API key is not valid UTF-8")?
            .trim()
            .to_owned();

        Ok((!api_key.is_empty()).then_some(api_key))
    })
}

pub fn save_openai_api_key(api_key: &str) -> Result<()> {
    let api_key = api_key.trim();
    if api_key.is_empty() {
        anyhow::bail!("API key cannot be empty");
    }

    future::block_on(async {
        let keyring = open_keyring().await?;
        keyring
            .create_item(
                "Scan Slicer OpenAI API key",
                &ATTRIBUTES,
                Secret::text(api_key),
                true,
            )
            .await
            .context("could not save the OpenAI API key")?;
        Ok(())
    })
}

pub fn delete_openai_api_key() -> Result<()> {
    future::block_on(async {
        let keyring = open_keyring().await?;
        keyring
            .delete(&ATTRIBUTES)
            .await
            .context("could not remove the OpenAI API key")?;
        Ok(())
    })
}

async fn open_keyring() -> Result<Keyring> {
    let keyring = Keyring::new()
        .await
        .context("could not open the system keyring")?;
    keyring
        .unlock()
        .await
        .context("could not unlock the system keyring")?;
    Ok(keyring)
}
