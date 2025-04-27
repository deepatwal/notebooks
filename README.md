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



refactor code to process n3 triples:

@prefix rdf: http://www.w3.org/1999/02/22-rdf-syntax-ns# .
@prefix owl: http://www.w3.org/2002/07/owl# .
@prefix rdfs: http://www.w3.org/2000/01/rdf-schema# .
@prefix xsd: http://www.w3.org/2001/XMLSchema# .
@prefix rdf4j: http://rdf4j.org/schema/rdf4j# .
@prefix sesame: http://www.openrdf.org/schema/sesame# .
@prefix fn: http://www.w3.org/2005/xpath-functions# .

http://dbpedia.org/resource/Volvo a owl:Thing, http://dbpedia.org/ontology/Organisation,
http://dbpedia.org/ontology/Company, http://dbpedia.org/ontology/Agent, http://schema.org/Organization,
http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Agent, http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#SocialPerson,
http://www.wikidata.org/entity/Q24229398, http://www.wikidata.org/entity/Q43229,
http://www.wikidata.org/entity/Q4830453;
http://xmlns.com/foaf/0.1/name "AB Volvo"@en;
http://dbpedia.org/ontology/numberOfEmployees "95850"^^xsd:nonNegativeInteger;
http://dbpedia.org/ontology/foundedBy http://dbpedia.org/resource/Assar_Gabrielsson,
http://dbpedia.org/resource/Gustav_Larson;
http://dbpedia.org/ontology/location http://dbpedia.org/resource/Gothenburg;
http://xmlns.com/foaf/0.1/homepage https://www.volvogroup.com/;
http://dbpedia.org/ontology/type http://dbpedia.org/resource/Aktiebolag;
http://dbpedia.org/ontology/keyPerson http://dbpedia.org/resource/Carl-Henric_Svanberg,
http://dbpedia.org/resource/Martin_Lundstedt;
http://dbpedia.org/ontology/industry http://dbpedia.org/resource/Automotive_industry;
http://dbpedia.org/ontology/product http://dbpedia.org/resource/Volvo_Trucks,
http://dbpedia.org/resource/Buses, http://dbpedia.org/resource/Construction_equipment;
http://dbpedia.org/ontology/subsidiary http://dbpedia.org/resource/Mack_Trucks,
http://dbpedia.org/resource/Renault_Trucks, http://dbpedia.org/resource/Volvo_Buses,
http://dbpedia.org/resource/Volvo_Construction_Equipment, http://dbpedia.org/resource/Volvo_Penta,
http://dbpedia.org/resource/Volvo_Trucks, http://dbpedia.org/resource/Arquus,
http://dbpedia.org/resource/Volvo_Financial_Services .

http://dbpedia.org/resource/1968_24_Hours_of_Le_Mans__DNQ__1 http://dbpedia.org/ontology/manufacturer
http://dbpedia.org/resource/Volvo .

http://dbpedia.org/resource/1970_12_Hours_of_Sebring__DNF__24 http://dbpedia.org/ontology/manufacturer
http://dbpedia.org/resource/Volvo .

http://dbpedia.org/resource/1970_12_Hours_of_Sebring__DNF__28 http://dbpedia.org/ontology/manufacturer
http://dbpedia.org/resource/Volvo .

http://dbpedia.org/resource/1970_24_Hours_of_Daytona__T2.0__3 http://dbpedia.org/ontology/manufacturer
http://dbpedia.org/resource/Volvo .

http://dbpedia.org/resource/1970_24_Hours_of_Daytona__T2.0__4 http://dbpedia.org/ontology/manufacturer
http://dbpedia.org/resource/Volvo .

http://dbpedia.org/resource/Volvo_B18_engine http://dbpedia.org/ontology/manufacturer
http://dbpedia.org/resource/Volvo .

http://dbpedia.org/resource/Volvo_B30_engine http://dbpedia.org/ontology/manufacturer
http://dbpedia.org/resource/Volvo .

http://dbpedia.org/resource/Volvo_B36_engine http://dbpedia.org/ontology/manufacturer
http://dbpedia.org/resource/Volvo .

http://dbpedia.org/resource/Volvo_Brage http://dbpedia.org/ontology/manufacturer
http://dbpedia.org/resource/Volvo;
http://dbpedia.org/ontology/engine http://dbpedia.org/resource/Volvo .

http://dbpedia.org/resource/Volvo_F85 http://dbpedia.org/ontology/manufacturer
http://dbpedia.org/resource/Volvo;
http://dbpedia.org/ontology/engine http://dbpedia.org/resource/Volvo .

http://dbpedia.org/resource/Volvo_F88 http://dbpedia.org/ontology/manufacturer
http://dbpedia.org/resource/Volvo;
http://dbpedia.org/ontology/engine http://dbpedia.org/resource/Volvo .

http://dbpedia.org/resource/Volvo_L340 http://dbpedia.org/ontology/manufacturer
http://dbpedia.org/resource/Volvo .

http://dbpedia.org/resource/Volvo_LV66-series http://dbpedia.org/ontology/manufacturer
http://dbpedia.org/resource/Volvo .

http://dbpedia.org/resource/Volvo_LV76-series http://dbpedia.org/ontology/manufacturer
http://dbpedia.org/resource/Volvo .

http://dbpedia.org/resource/Volvo_LV81-series http://dbpedia.org/ontology/manufacturer
http://dbpedia.org/resource/Volvo;
http://dbpedia.org/ontology/engine http://dbpedia.org/resource/Volvo .

http://dbpedia.org/resource/Volvo_Longnose http://dbpedia.org/ontology/manufacturer
http://dbpedia.org/resource/Volvo;
http://dbpedia.org/ontology/engine http://dbpedia.org/resource/Volvo .

http://dbpedia.org/resource/Volvo_Roundnose http://dbpedia.org/ontology/manufacturer
http://dbpedia.org/resource/Volvo;
http://dbpedia.org/ontology/engine http://dbpedia.org/resource/Volvo .

http://dbpedia.org/resource/Volvo_Titan http://dbpedia.org/ontology/manufacturer
http://dbpedia.org/resource/Volvo;
http://dbpedia.org/ontology/engine http://dbpedia.org/resource/Volvo .

http://dbpedia.org/resource/Volvo_Viking http://dbpedia.org/ontology/manufacturer
http://dbpedia.org/resource/Volvo;
http://dbpedia.org/ontology/engine http://dbpedia.org/resource/Volvo .

http://dbpedia.org/resource/White_Motor_Company http://dbpedia.org/ontology/successor
http://dbpedia.org/resource/Volvo .

http://dbpedia.org/resource/Carl-Henric_Svanberg http://dbpedia.org/ontology/occupation
http://dbpedia.org/resource/Volvo .

http://dbpedia.org/resource/Nova_Bus http://dbpedia.org/ontology/owningCompany
http://dbpedia.org/resource/Volvo;
http://dbpedia.org/ontology/owner http://dbpedia.org/resource/Volvo .

http://dbpedia.org/resource/Assar_Gabrielsson http://dbpedia.org/ontology/knownFor
http://dbpedia.org/resource/Volvo .

http://dbpedia.org/resource/Gustaf_Larson http://dbpedia.org/ontology/knownFor
http://dbpedia.org/resource/Volvo .

http://dbpedia.org/resource/Jan_G._Smith http://dbpedia.org/ontology/knownFor
http://dbpedia.org/resource/Volvo .

http://dbpedia.org/resource/Mack_Trucks http://dbpedia.org/ontology/parentCompany
http://dbpedia.org/resource/Volvo .

http://dbpedia.org/resource/Renault_Trucks http://dbpedia.org/ontology/parentCompany
http://dbpedia.org/resource/Volvo .

http://dbpedia.org/resource/SDLG http://dbpedia.org/ontology/parentCompany http://dbpedia.org/resource/Volvo .

http://dbpedia.org/resource/Volvo_Aero http://dbpedia.org/ontology/parentCompany
http://dbpedia.org/resource/Volvo .

http://dbpedia.org/resource/Volvo_Buses http://dbpedia.org/ontology/parentCompany
http://dbpedia.org/resource/Volvo .

http://dbpedia.org/resource/Volvo_Construction_Equipment http://dbpedia.org/ontology/parentCompany
http://dbpedia.org/resource/Volvo .

http://dbpedia.org/resource/Volvo_Trucks http://dbpedia.org/ontology/parentCompany
http://dbpedia.org/resource/Volvo .

http://dbpedia.org/resource/Alexander_Dennis_Enviro500 http://dbpedia.org/ontology/engine
http://dbpedia.org/resource/Volvo .

http://dbpedia.org/resource/Volvo_B7L http://dbpedia.org/ontology/engine http://dbpedia.org/resource/Volvo .

http://dbpedia.org/resource/Volvo_VN http://dbpedia.org/ontology/engine http://dbpedia.org/resource/Volvo .

http://dbpedia.org/resource/Nils_Bohlin http://dbpedia.org/ontology/employer http://dbpedia.org/resource/Volvo .

http://dbpedia.org/resource/Pehr_G._Gyllenhammar__Pehr_G._Gyllenhammar__1 http://dbpedia.org/ontology/employer
http://dbpedia.org/resource/Volvo .

http://dbpedia.org/resource/Selam_Bus_Line_Share_Company http://dbpedia.org/ontology/vehiclesInFleet
http://dbpedia.org/resource/Volvo .

to generate output like:

IRI: http://dbpedia.org/resource/Volvo
Outgoing:
label: Volvo
name: "AB Volvo"@en;
type: [Organisation, Company, Agent]
numberOfEmployees: 95850
foundedBy: Assar Gabrielsson
Incoming:
(1968 24 Hours of Le Mans DNQ 1, manufacturer, Volvo)
(Volvo B18 engine, manufacturer, Volvo)
(Volvo F88, engine, Volvo)


props: defaultdict(<class 'list'>, {'type': ['Organisation', 'Agent', 'Q4830453', 'Q43229', 'SocialPerson', 'Thing', 'Q24229398', 'Aktiebolag', 'Agent', 'Company', 'Organization'], 'name': ['AB Volvo'], 'keyPerson': ['Martin_Lundstedt', 'Carl-Henric_Svanberg'], 'subsidiary': ['Volvo_Penta', 'Volvo_Construction_Equipment', 'Mack_Trucks', 'Renault_Trucks', 'Volvo_Buses', 'Volvo_Financial_Services', 'Volvo_Trucks', 'Arquus'], 'location': ['Gothenburg'], 'foundedBy': ['Gustav_Larson', 'Assar_Gabrielsson'], 'product': ['Buses', 'Volvo_Trucks', 'Construction_equipment'], 'numberOfEmployees': ['95850'], 'homepage': [''], 'industry': ['Automotive_industry']})
2025-04-27 18:56:27,958 [INFO] incoming: [('Volvo_Titan', 'engine', 'Volvo'), ('White_Motor_Company', 'successor', 'Volvo'), ('Volvo_F88', 'engine', 'Volvo'), ('Alexander_Dennis_Enviro500', 'engine', 'Volvo'), ('Volvo_VN', 'engine', 'Volvo'), ('SDLG', 'parentCompany', 'Volvo'), ('1970_24_Hours_of_Daytona__T2.0__4', 'manufacturer', 'Volvo'), ('Volvo_B30_engine', 'manufacturer', 'Volvo'), ('Volvo_F85', 'engine', 'Volvo'), ('Volvo_LV66-series', 'manufacturer', 'Volvo'), ('Volvo_Buses', 'parentCompany', 'Volvo'), ('Volvo_Roundnose', 'engine', 'Volvo'), ('Volvo_Longnose', 'engine', 'Volvo'), ('Volvo_Brage', 'manufacturer', 'Volvo'), ('Pehr_G._Gyllenhammar__Pehr_G._Gyllenhammar__1', 'employer', 'Volvo'), ('1970_12_Hours_of_Sebring__DNF__24', 'manufacturer', 'Volvo'), ('Mack_Trucks', 'parentCompany', 'Volvo'), ('Volvo_B36_engine', 'manufacturer', 'Volvo'), ('1970_24_Hours_of_Daytona__T2.0__3', 'manufacturer', 'Volvo'), ('Nova_Bus', 'owningCompany', 'Volvo'), ('Volvo_LV81-series', 'engine', 'Volvo'), ('Volvo_F85', 'manufacturer', 'Volvo'), ('Volvo_Viking', 'manufacturer', 'Volvo'), ('Volvo_Trucks', 'parentCompany', 'Volvo'), ('Selam_Bus_Line_Share_Company', 'vehiclesInFleet', 'Volvo'), ('Volvo_L340', 'manufacturer', 'Volvo'), ('Nova_Bus', 'owner', 'Volvo'), ('Jan_G._Smith', 'knownFor', 'Volvo'), ('1968_24_Hours_of_Le_Mans__DNQ__1', 'manufacturer', 'Volvo'), ('Volvo_Roundnose', 'manufacturer', 'Volvo'), ('Volvo_Longnose', 'manufacturer', 'Volvo'), ('Nils_Bohlin', 'employer', 'Volvo'), ('Volvo_B7L', 'engine', 'Volvo'), ('Volvo_B18_engine', 'manufacturer', 'Volvo'), ('Volvo_Viking', 'engine', 'Volvo'), ('Renault_Trucks', 'parentCompany', 'Volvo'), ('1970_12_Hours_of_Sebring__DNF__28', 'manufacturer', 'Volvo'), ('Volvo_Titan', 'manufacturer', 'Volvo'), ('Volvo_Construction_Equipment', 'parentCompany', 'Volvo'), ('Volvo_F88', 'manufacturer', 'Volvo'), ('Volvo_LV81-series', 'manufacturer', 'Volvo'), ('Gustaf_Larson', 'knownFor', 'Volvo'), ('Volvo_LV76-series', 'manufacturer', 'Volvo'), ('Carl-Henric_Svanberg', 'occupation', 'Volvo'), ('Assar_Gabrielsson', 'knownFor', 'Volvo'), ('Volvo_Brage', 'engine', 'Volvo'), ('Volvo_Aero', 'parentCompany', 'Volvo')]
