# MLB Tool Grades

Can't run without adding a statcast.db file but too large of a file to worry about adding. You could also add one or collection of csv's of statcast data in the data/statcast/ folder

## Examples:

### Build Models
`python train.py`

### Create Grades
`python get_grades.py --suffix=2024Grades --start_date=2024-04-01 --end_date=2024-12-31 --year=2024`
