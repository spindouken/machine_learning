# Advanced Linear Algebra From Scratch

## Project Summary

This project focuses on advanced linear algebra operations while utilizing no libraries, including the calculation of determinants, minors, cofactors, adjugates, inverses, and definiteness of matrices. It emphasizes handling input validation and correctly identifying properties of square matrices, ensuring robustness and reliability in mathematical computations.

### Task Summaries

1. **Determinant**: 
   - Implements a function to calculate the determinant of a square matrix. It includes validation checks for input type and matrix dimensions, returning the determinant or raising errors for invalid inputs.

2. **Minor**: 
   - Develops a function to compute the minor matrix of a given square matrix. It ensures the input is a valid list of lists and raises errors for non-square or empty matrices.

3. **Cofactor**: 
   - Creates a function to calculate the cofactor matrix for a square matrix, incorporating similar input validation as the previous tasks to ensure proper matrix format and dimensions.

4. **Adjugate**: 
   - Writes a function to compute the adjugate matrix of a square matrix, maintaining the same validation requirements to check for input type and matrix properties.

5. **Inverse**: 
   - Implements a function to calculate the inverse of a square matrix, handling input validation and returning `None` for singular matrices that do not have an inverse.

6. **Definiteness**: 
   - Adds a function to determine the definiteness of a matrix using NumPy, with validation checks for the input type and returning specific categories based on the matrix properties.

