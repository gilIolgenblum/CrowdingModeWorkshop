# FH Crowding Research Tool

A thermodynamic mean-field model for protein folding in binary and ternary crowding mixtures.

## Installation

You can install this package in editable mode using `pip`:

```bash
pip install -e .
```

To install with development dependencies (e.g., Jupyter for running notebooks):
```bash
pip install -e .[dev]
```

To install with app dependencies (e.g., Streamlit for the local web app):
```bash
pip install -e .[app]
```

## Tutorials

We provide several user-facing Jupyter notebook tutorials in the `tutorials/` directory to help you learn the Python API:

1. `00_getting_started.ipynb`: Installation, scientific background, and setup.
2. `01_binary_plug_and_play_simulation.ipynb`: Running forward binary simulations.
3. `02_binary_fit_experimental_data.ipynb`: Fitting soft interaction parameters to binary data.
4. `03_ternary_plug_and_play_simulation.ipynb`: Running forward simulations for 2-cosolute mixtures.
5. `04_ternary_fit_experimental_data.ipynb`: Fitting soft interaction parameters to ternary 2D data grids.
6. `05_parameter_sensitivity_and_interpretation.ipynb`: Understanding the physical effects of SASA and Flory-Huggins $\chi$.
7. `06_exporting_results_and_figures.ipynb`: Extracting datasets and tweaking matplotlib charts for publication.

Start with `tutorials/00_getting_started.ipynb`!

Note: While the Streamlit app is available for non-coding users, these notebooks are intended for reproducible scripted analysis.

## Minimal Working Example

```python
from fh_crowding import Protein, Cosolute, BinaryCrowdingModel

# Define protein and cosolute parameters
protein = Protein(SASA=419.0)
urea = Cosolute(nu=1.0, chi=0.1, chiTS=-0.05)

# Initialize the model
model = BinaryCrowdingModel(
    protein=protein,
    cosolute=urea,
    eps=0.0,
    epsTS=0.0,
    T=298.15
)

# Solve equilibrium
model.solve_equil()

# Export results to DataFrame
results = model.to_pandas()
print(results.head())
```

## Assumptions

This tool implements a mean-field model of Flory-Huggins type, accounting for excluded volume (nu), non-ideal mixing (chi, chiTS), and soft interaction (eps, epsTS) effects around a protein domain.

## License
MIT License

## Citation

If you use this software or Streamlit app, please cite the associated publications and the software repository.

Software repository: https://github.com/gilIolgenblum/ProteinCrowding

Developer: Gil I. Olgenblum  
ORCID: https://orcid.org/0000-0002-4514-5516

A versioned software DOI will be added after archival release.

### References

1. Sapir, L.; Harries, D. Macromolecular Stabilization by Excluded Cosolutes: Mean Field Theory of Crowded Solutions. *Journal of Chemical Theory and Computation* **2015**, *11* (7), 3478-3490.
2. Sapir, L.; Harries, D. Macromolecular Compaction by Mixed Solutions: Bridging versus Depletion Attraction. *Current Opinion in Colloid & Interface Science* **2016**, *22*, 80-87.
3. Stewart, C. J.; Olgenblum, G. I.; Propst, A.; Harries, D.; Pielak, G. J. Resolving the Enthalpy of Protein Stabilization by Macromolecular Crowding. *Protein Science* **2023**, *32* (3), e4573.
4. Olgenblum, G. I.; Carmon, N.; Harries, D. Not Always Sticky: Specificity of Protein Stabilization by Sugars Is Conferred by Protein-Water Hydrogen Bonds. *Journal of the American Chemical Society* **2023**, *145* (42), 23308-23320.
5. Olgenblum, G. I.; Stewart, C. J.; Redvanly, T. W.; Young, O. M.; Lauzier, F.; Hazlett, S.; et al.; Harries, D. Crowding beyond Excluded Volume: A Tale of Two Dimers. *Protein Science* **2025**, *34* (4), e70062.
6. Redvanly, T. W.; Olgenblum, G. I.; Young, O. M.; Goldenberg, Y.; Stewart, C. J.; Harries, D.; Pielak, G. J. Sugar-Protein Interactions Control Protein-Complex Stability in Crowded Ficoll and Dextran Solutions. *Protein Science* **2026**, *35* (1), e70416.
