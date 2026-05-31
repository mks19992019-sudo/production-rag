from app.extractor import extract
from app.graph import get_workflow





def test_extractor():
    data = extract()
    print(data)
    assert data is not None
    assert len(data) > 0



from app.extractor import extract

docs = extract("https://cpur.in")

#print(docs)