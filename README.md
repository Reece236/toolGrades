# MLB Tool Grades

This project predicts and evaluates player hitting tools using pitch–by–pitch Statcast data.  LightGBM models estimate swing decisions, contact ability and exit velocity, while Bayesian methods quantify uncertainty around those grades.  An overall projection model combines the posterior samples to forecast future production.  The pipeline pulls data from a SQLite database, trains models with hyperparameter optimization, and generates 20–80 scaled grades with credible intervals.

The codebase demonstrates production best practices such as modular data pipelines, reproducible model training, and CLI entry points.  Models and features are persisted for reuse, and the build script can recreate the database from raw CSVs.  The project highlights skills in Python, SQL, machine learning and probabilistic modeling that align closely with the Dodgers’ analytics workflow.

## Usage

```bash
# Train component models
python main.py train

# Train overall projection model
python main.py overall

# Generate grades (requires trained models and Statcast DB)
python main.py grades --suffix=2024Grades --start_date=2024-04-01 --end_date=2024-12-31 --year=2024
```
