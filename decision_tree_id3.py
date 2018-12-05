training_data = [
    ['Green', 3, 'Mango'],
    ['Yellow', 3, 'Mango'],
    ['Red', 1, 'Grape'],
    ['Red', 1, 'Grape'],
    ['Yellow', 3, 'Lemon']
]

# Column labels
# Used only to print the tree
header = ['color', 'diameter', 'label']

def unique_vals(rows, cols):
    # Find unique values for a column in a dataset
    return set([row[col] for row in rows])

def class_counts(rows):
    # Counts the no of each type of example in a dataset
    counts = {}
    for row in rows:
        label = row[-1] # The last column is the label
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

def is_numeric(value):
    return isinstance(value, int) or isinstance(value, float)

class Question:
    """
    A Question is used to partition a dataset.
    
    This class just records a 'column number' (e.g., 0 for color) and a
    'column value' (e.g., Green). The 'match' method is used to compare
    the feature value in an example to the feature value stored in the
    question.
    """
    
    def __init__(self, column, value):
        self.column = column
        self.value = value
        
    def match(self, example):
        # Compare the feature value in an example to the feature value in this question
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        
    def __repr__(self):
        # Helper to print the question in a readable format
        condition = '=='
        if is_numeric(self.value):
            condition = '>='
        return 'Is %s %s %s?'%(header[self.column], condition, str(self.value))
    
def partition(rows, question):
    """
    Partitions a dataset.
    For each row in the dataset, check if it matches the question.
    If so, add it to the 'true rows', otherwise to 'false rows'.
    """

    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)

    return true_rows, false_rows

def gini(rows):
# Calculate Gini Impurity for a list of rows.

    counts = class_counts(rows)
    impurity = 1
    for label in counts:
        prob_label = counts[label] / float(len(rows))
        impurity -= prob_label ** 2

    return impurity

def info_gain(left, right, current_uncertainty):
    """
    Information Gain.
    The uncertainty of the starting node minus the weighted impurity of 
    the two child nodes.
    """

    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)
    
def find_best_split(rows):
    """
    Find the best question to ask by iterating over every feature/value
    and calculation information gain.
    """

    best_gain = 0
    best_question = None
    current_uncertainty = gini(rows)
    n_features = len(rows[0]) - 1 # no of columns

    for col in range(n_features): # for each feature

        values = set([row[col] for row in rows]) # unique values in the column

        for val in values:
            question = Question(col, val)

            # Try splitting the dataset
            true_rows, false_rows = partition(rows, question)

            # Skip this split if it doesn't divide the dataset
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue;

            # Calculate the information gain from this split
            gain = info_gain(true_rows, false_rows, current_uncertainty)

            if gain >= best_gain:
                best_gain, best_question = gain, question

    return best_gain, best_question
    
class Leaf:
    """ A leaf node classifies data.
    This holds a dictionary of class(e.g., "Mango") -> number of times it
    appears in the rows from the training data that reach this leaf.
    """
    
    def __init__(self, rows):
        self.predictions = class_counts(rows)
        

class Decision_Node:
    """ A Decision node asks a question.
    Holds a reference to the question and to the two child nodes.
    """
    
    def __init__(self, question, true_branch, false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch
        
def build_tree(rows):
    
    gain, question = find_best_split(rows)
    
    # Base case: no further info gain, return leaf
    if gain == 0:
        return Leaf(rows)
    
    # We have found a useful feature/value to partition on
    true_rows, false_rows = partition(rows, question)
    
    # Recursively build the true branch.
    true_branch = build_tree(true_rows)
    
    # Recursively build the false branch.
    false_branch = build_tree(false_rows)
    
    # Returns a Question node. Records the best feature/value to ask at this point
    return Decision_Node(question, true_branch, false_branch)

def print_tree(node, spacing=""):
    
    # Base case: We've reached a leaf
    if isinstance(node, Leaf):
        print(spacing + "Predict", node.predictions)
        return
    
    # Print the question at this node
    print(spacing + str(node.question))
    
    # Call this f() recursively on the true branch
    print(spacing + '--> True:')
    print_tree(node.true_branch, spacing + ' ')
    
    # Call this f() recursively on the false branch
    print(spacing + '--> False:')
    print_tree(node.false_branch, spacing + ' ')
    
def classify(row, node):
    # Base case: We've reached a leaf
    if isinstance(node, Leaf):
        return node.predictions
    
    # Decide whether to follow true or false branch.
    # Compare the feature/value stored in the node to the considered example
    
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)

def print_leaf(counts):
    # Print the predictions at a leaf
    total = sum(counts.values()) * 1.0
    probs = {}
    
    for label in counts.keys():
        probs[label] = str(int(counts[label] / total * 100)) + '%'
        
    return probs

if __name__ == '__main__':
    
    my_tree = build_tree(training_data)
    
    print_tree(my_tree)
    
    testing_data = [
        ['Green', 3, 'Apple'],
        ['Yellow', 4, 'Apple'],
        ['Red', 2, 'Grape'],
        ['Red', 1, 'Grape'],
        ['Yellow', 3, 'Lemon'],
    ]
    
    for row in testing_data:
        print('Actual: %s. Predicted: %s' %(row[-1], print_leaf(classify(row, my_tree))))              