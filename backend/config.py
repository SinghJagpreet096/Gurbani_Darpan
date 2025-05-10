from dataclasses import dataclass

@dataclass
class Config:
    model: str = 'llama3.2'
    database_dir: str = 'backend/data/'
    database_path: str = f"{database_dir}mydata.sqlite"
    max_angs: int = 1430
    chunk_size: int = max_angs // 10
    
    api_url: str = 'https://api.banidb.com/v2/angs'
    source_id: str = 'G' ## Guru Granth Sahib Ji