[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
## SYRUP: Knowledge Capturing using SYmbolic RUles by detecting SYnonym Predicates

SYRUP is a two-fold approach to detect synonym predicates and alternative definitions based on symbolic rules; it defines a metric to measure whether synonym predicates are complementary. These methods are implemented in the engine SYRUP. The experimental results depict that detecting synonym predicates and alternative definitions based on symbolic approaches are capable of competing with the state-of-the-art embedding-based techniques. To show the effectiveness of SYRUP, the discovered synonym predicates and alternative definition of predicates are used in query expansion to retrieve complete answers. 

![SYRUP example](/images/MotivatingExample.png?raw=true "SYRUP example")





### Building SYRUP from Source
Clone the repository
```git
git clone git@github.com:SDM-TIB/SYRUP.git
```
```python
pip install -r requirements.txt
```
Configuration for executing
```json
{
  "KG": "DBpedia",
  "endpoint": "https://labs.tib.eu/sdm/dbpedia/sparql",
  "prefix": "http://dbpedia.org/ontology/",
  "rules_file": "DBpedia_AMIE_Rules.csv",
  "domain": "Person",
  "query": "Q1"
}
```

The proposed approach is a a knowledge graph-agnostic approach. Therefore, apart from DBpedia, other KGs, i.e., ```FrenchRoyalty``` or ```Family``` can be used as the parameter ``KG``.
```json
{
  "KG": "Family",
  "endpoint": "https://labs.tib.eu/sdm/family_kg/sparql",
  "prefix": "http://family.org/",
  "rules_file": "Family_AMIE_Rules.csv",
  "domain": "Family",
  "query": "Q1"
}
```
```python
python SYRUP.py 
```
### Plots demonstrating
```python
python -m plot_evaluation -e experiments/DBpedia
```

![SYRUP evaluation](/images/PrecisionRecallEval.png?raw=true "SYRUP evaluation")
