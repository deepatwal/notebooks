# notebooks

# ollama

- model path: /usr/share/ollama/.ollama/models/blobs/
- sudo systemctl status ollama.service
- sudo systemctl start ollama.service
- sudo systemctl stop ollama.service

Loading data into a repository created from Workbench:
    $ <graphdb-dist>/bin/importrdf load -f -i <repo-id> -m parallel <RDF data file(s)>

        cd -> /c/Users/deepa/data/softwares/graphdb/graphdb-11.0.0-dist/graphdb-11.0.0/bin
        ./importrdf load -f -i dbpedia-14-04-2025-No-Inference -m parallel /c/Users/deepa/graphdb-import/dbpedia-14-04-2025


        cd -> C:\Users\deepa\AppData\Local\GraphDB Desktop\app\lib
            java -cp "*" com.ontotext.graphdb.importrdf.ImportRDF load -f -i dbpedia-14-04-2025-No-Inference -m parallel /c/Users/deepa/graphdb-import/dbpedia-14-04-2025

