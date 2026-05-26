from app.extractor import extract



def test_extractor():
    data = extract()
    print(data)
    assert data is not None
    assert len(data) > 0



from app.extractor import extract

docs = extract("https://cpur.in")

print(docs)