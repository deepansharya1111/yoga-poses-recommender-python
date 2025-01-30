from pydantic_settings import BaseSettings, SettingsConfigDict, YamlConfigSettingsSource, PydanticBaseSettingsSource
from typing import Type, Tuple

class Settings(BaseSettings):
    project_id: str
    location: str
    gemini_model_name: str
    embedding_model_name: str
    image_generation_model_name: str
    database: str
    collection: str
    test_collection: str
    top_k: int

    model_config = SettingsConfigDict(
        yaml_file="config.yaml", yaml_file_encoding="utf-8"
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return (YamlConfigSettingsSource(settings_cls),)
    
def get_settings() -> Settings:
    return Settings()
