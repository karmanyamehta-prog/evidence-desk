from dataclasses import dataclass
from pydantic import BaseModel, Field

@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    metadata: dict

@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    distance: float

class Evidence(BaseModel):
    source_id: str
    quote: str = Field(min_length=12)

class Claim(BaseModel):
    text: str = Field(min_length=1)
    evidence: list[Evidence] = Field(min_length=1)

class Draft(BaseModel):
    abstain: bool = False
    claims: list[Claim] = Field(default_factory=list)

class Query(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    document_ids: list[str] = Field(default_factory=list, max_length=20)
