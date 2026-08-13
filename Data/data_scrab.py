from langchain_community.document_loaders import WebBaseLoader
import logfire
import json
from pathlib import Path


URL = "https://www.cpur.in"

logfire.configure()
logfire.instrument_system_metrics()
loder = WebBaseLoader(URL)
logfire.info('web data is succesfully load')
data =loder.load()


data = "\n\n".join(doc.page_content for doc in data)

output_dir = Path("DATA/raw")

output_dir.mkdir(parents=True,exist_ok=True)

with open(output_dir / "cpur_data.txt" , "w" , encoding="utf-8") as f:
    f.write(data)

logfire.info(f'data is store in text formate in dir {output_dir}')

