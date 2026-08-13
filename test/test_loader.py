from src.dataInjection.data_injection import extract






def test_extractor():
    data = extract()
    print(data)
    assert data is not None
    assert len(data) > 0





docs = extract()

print(docs)