from dataclasses import dataclass


@dataclass(frozen=True)
class DataIngestionArtifact:
    raw_data_path: str


@dataclass(frozen=True)
class DataValidationArtifact:
    validation_status: bool