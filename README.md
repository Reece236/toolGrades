# MLB Tool Grades

MLB Tool Grades scores hitters on the 20–80 scouting scale using Statcast pitch data.  LightGBM models first predict swing decision, strike probability, batted-ball outcomes and exit velocity.  Hierarchical Bayesian models then pool information across players to estimate underlying skill and produce credible intervals for each grade.  A separate overall model uses these posterior samples and sprint speed to forecast future production.

The code illustrates production practices such as reproducible pipelines, hyperparameter tuning and saved model artifacts.  Data can be ingested from a SQLite database or rebuilt from raw CSV files.  Command line entry points manage training, overall projections and grade generation.  The project demonstrates expertise in Python, SQL and modern Bayesian modeling—skills directly applicable to a professional baseball analytics group.

## Usage

```bash
# Train component models
python main.py train

# Train overall projection model
python main.py overall

# Generate grades (requires trained models and Statcast DB)
python main.py grades --suffix=2024Grades --start_date=2024-04-01 --end_date=2024-12-31 --year=2024
```
