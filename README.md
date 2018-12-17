__sudoku_genetic_python__ is a Sudoku solver using Python and genetic algorithm which is implemented using this [paper](./paper.pdf).

# Usage
```
$ pythpn3 sudoku_genetic_python --help
usage: suduko_genetic_python.py [-h] [-o OUTPUT_FILE] [-p POPULATION_SIZE]
                                [-s SELECTION_RATE] [-m MAX_GENERATIONS_COUNT]
                                [-u MUTATION_RATE] [-q]
                                file

positional arguments:
  file                  Input file that contains Sudoku's problem.

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT_FILE, --output-file OUTPUT_FILE
                        Output file to store problem's solution.
  -p POPULATION_SIZE, --population-size POPULATION_SIZE
  -s SELECTION_RATE, --selection-rate SELECTION_RATE
  -m MAX_GENERATIONS_COUNT, --max-generations-count MAX_GENERATIONS_COUNT
  -u MUTATION_RATE, --mutation-rate MUTATION_RATE
  -q, --quiet
```

## Author

Hamidreza Mahdavipanah

## License

[MIT](./LICENSE)
