import glob
import re
from langchain_core.documents import Document
import logfire

logfire.configure()
logfire.instrument_system_metrics()


# use glob for searching 
# it return the complete list 

#path = glob.glob('Data/campasX_data/*.vtt')

#print(path)

#lines=[]

#for line in open(path[0]):
    #print(line)
    #line = line.strip()
    #if not line or line == "WEBVTT" or '-->' in line:
        #continue
    #lines.append(line)


#text = " ".join(lines)
#print(text)



def load_transcript():
    logfire.info('Loading the data.... from raw dataset')


    docs = []

    for path in glob.glob("Data/campasX_data/*.vtt"):
        lines = []
        for line in open(path):
            line = line.strip()

            if not line or line == "WEBVTT" or "-->" in line:
                continue
            lines.append(line)
        text = " ".join(lines)
        session = re.search(r"Session[ _]*(\d+)", path).group(1)
        docs.append(Document(page_content=text, metadata={"session": session}))



    return docs







        




    



