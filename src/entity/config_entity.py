from dataclasses import dataclass


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: str
    source_file: str
    raw_data_path: str


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: str
    STATUS_FILE: str