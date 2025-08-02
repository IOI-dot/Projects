import numpy as np
import sys
def rref(matrix):
    A= matrix.astype(float)
    rows, cols = A.shape
    pivot_row = 0
    # FORWARD phase: convert to row echelon form
    for col in range(cols):
        pivot = None
        for row in range(pivot_row, rows):
            if A[row, col] != 0:
                pivot = row
                break
        if pivot is None:
            continue

        if pivot != pivot_row:
            A[[pivot_row, pivot]] = A[[pivot, pivot_row]]

        A[pivot_row] = A[pivot_row] / A[pivot_row, col]

        for r in range(pivot_row + 1, rows):
            A[r] = A[r] - A[r, col] * A[pivot_row]

        pivot_row += 1
        if pivot_row == rows:
            break
    # BACKWARD phase: make above pivots 0
    for col in reversed(range(cols)):
        pivot_row = None
        for row in reversed(range(rows)):
            if np.isclose(A[row, col], 1) and all(np.isclose(A[row, :col], 0)):
                pivot_row = row
                break
        if pivot_row is not None:
            for r in range(pivot_row):
                A[r] = A[r] - A[r, col] * A[pivot_row]
    return A

def determinant(matrix):
    A = matrix.astype(float)
    n = A.shape[0]
    det = 1
    for i in range(n):
        pivot = None
        for j in range(i, n):
            if A[j, i] != 0:
                pivot = j
                break
        if pivot is None:
            return 0
        if pivot != i:
            A[[i, pivot]] = A[[pivot, i]]
            det *= -1

        det *= A[i, i]
        A[i] = A[i] / A[i, i]
        for j in range(i + 1, n):
            A[j] -= A[j, i] * A[i]
    return round(det, 5)

def inverse(matrix):
    n = matrix.shape[0]
    A = matrix.astype(float)
    I = np.eye(n)
    AI = np.hstack((A, I))

    for i in range(n):
        pivot = None
        for j in range(i, n):
            if AI[j, i] != 0:
                pivot = j
                break
        if pivot is None:
            return None
        if pivot != i:
            AI[[i, pivot]] = AI[[pivot, i]]

        AI[i] = AI[i] / AI[i, i]
        for j in range(n):
            if j != i:
                AI[j] -= AI[j, i] * AI[i]

    left = AI[:, :n]
    right = AI[:, n:]
    if np.allclose(left, np.eye(n)):
        return right
    else:
        return None

def print_matrix(matrix, decimals=5):
    print("[")
    for row in matrix:
        formatted_row = []
        for val in row:
            val = 0 if abs(val) < 1e-10 else val  # round tiny numbers to zero
            formatted_row.append(f"{val:.{decimals}f}")
        print(" ".join(formatted_row))
    print("]")
# Main Program
if __name__ == "__main__":
    print("Enter a 4x4 matrix M row by row (each row must have 4 numbers, separated by spaces):")
    A = []
    for i in range(4):
        row_input = input(f"Row {i+1}: ").split()
        if len(row_input) != 4:
            print("Error: Each row must have exactly 4 numbers.")
            sys.exit(1)
        try:
            row = list(map(float, row_input))
            A.append(row)
        except ValueError:
            print("Error: Invalid number entered.")
            sys.exit(1)

    A = np.array(A)
    if A.shape != (4, 4):
        print("Error: Matrix must be 4x4.")
        sys.exit(1)

    print("\nOriginal Matrix M:")
    print_matrix(A)

    rref_A = rref(A.copy())
    print("\nOutput 1: RREF of M:")
    print_matrix(rref_A)

    det_A = determinant(A.copy())
    print(f"\nOutput 2: Determinant of M: {det_A}")

    inv_A = inverse(A.copy())
    print("\nOutput 3: Inverse of M:")
    if inv_A is not None:
        print_matrix(inv_A)
    else:
        print("Matrix M is not invertible.")
