# bc-praca

## Description
This repository contains the code for the bachelor's thesis in Data Science. The `apportionment.py` file contains backend code that encapsulates all necessary methods and classes for apportionment methods, data generating algorithms with bootstrap, and methods that help transform data into visualizable content. This structure is designed to keep Jupyter notebooks concise and focused.

Synthetic generated is not available as part of this electronic attachement due to its size, which amounts to hundreds of gigabytes. Nevertheless, it could be generated using our code, though the expected computation time spans over a few days. All data needed for visualisation, are, however, available in this attachement. All the code to generate this type of data is commented, because it would take tens of minutes up to several hours to complete.

Note that the data were generated randomly in 500 iterations and averaged. Every generated data might be a bit different. 

## Structure
- `apportionment.py`: Contains backend code for apportionment methods, data generating algorithms, and data transformation methods.
- `constants.py`: Defines constants used throughout the project.
- `question_0.ipynb` to `question_5.ipynb`: Jupyter notebooks for answering specific research questions in Chapter 3.
- `section_2_3.ipynb` and `section_2_4.ipynb`: Jupyter notebooks for detailed analysis sections 2.3 and 2.4.
- `real_data`: Folder containing real data used in the analysis.
- `vis_data`, `db_exports` and `db_exports_specific`: Folder containing processed synthetic data used in the analysis.


## Installation and Dependencies
This project requires Python 3.8 or newer installed.
You can install the required libraries using the following commands:
 `pip install [name_of_library]` or `conda install [name_of_library]` depending on your Python distribution.

## Documentation
While there is a reasonable amount of commentary inside our code, you can find more detailed explanations of the methods in the thesis.
