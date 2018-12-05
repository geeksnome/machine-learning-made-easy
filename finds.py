import csv
with open('tennis1.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)

h = ['0', '0', '0', '0', '0', '0']

for row in data:
    if row[-1] == 'True':
        j = 0
        
        for col in row:
            if col != 'True':
                if col != h[j] and h[j] == '0':
                    h[j] = col
                elif col != h[j] and h[j] != '0':
                    h[j] = '?'
                    
            j = j + 1
    
print('Maximally Specific Hypothesis: ', h)