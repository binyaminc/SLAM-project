import part1
import part2

database = part2.Database('data - select_better_PnP_or_P3P.pkl')
pairs = database.Pairs
tracks = database.Tracks

print(len(pairs))