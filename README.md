# **Dataless Neural Network Models For Variants of Domination**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Authors**: [Azib Farooq](https://github.com/itsazibfarooq), [Sangram K. Jena](https://sites.google.com/view/sangramkishorjena/home)  
**Institution**: Miami University  
**Submitted In**: [CPAIOR], [2025]

---

## **Abstract**
A Disjunctive Dominating Set (DDS) of graph G(V, E) is a set D,
such that v ∈ V (G)\D is either adjacent to u ∈ D or at distance two from the
two vertices u, w ∈ D. The objective of the DDS is to find minimum cardinality
in a graph G(V, E). This problem is recognized in the literature and known to
be NP-hard. In this paper, we explore DDS problem and one of its variant
i.e. Liar’s Dominating Set (LDS) problem, which tries to find a set L ⊆ V of a
graph G = (V, E) such that (1) for every vertex u ∈ V , |N [u] ∩ L| ≥ 2 and (2)
for every pair (u, v) ∈ V of distinct vertices, |(N [u] ∪ N [v]) ∩ L| ≥ 3. Drawing
inspiration from the dataless Neural Network (dNN), we formulate these two
problems in differentiable representation and design dNNs for them. We have
also provided pytorch implementation and evaluation for DDS problem, along
with proof of correctness.

---

## **Repository Overview**

This repository contains:  
- **models**: contain files for the dNN architecture.  
- **mdds_graphs**: 100 random graphs with small number of vertices and variable edge density.  
- **Solver**: Training Loops and driver code making use of dNN models.  

---

## **Getting Started**

### **Prerequisites**
- Python 3.x (or specify language/environment versions)
- Required libraries:  
  ```bash
  git clone https://github.com/itsazibfarooq/dNN
  cd dNN
  pip install -r requirements.txt
  ```


## **Citing This Work**

If you find our work useful, please consider citing it in your research or projects. Use the following BibTeX entry:

```bibtex
@article{author2024,
  title={Dataless Neural Network Models For Variants of Domination},
  author={Azib Farooq and Sangram K. Jena},
  journal={CPAIOR},
  year={2025}
}
