import numpy as np
import scipy
import scipy.linalg as la
from tkinter import *

# This is a simple matrix calculator that was designed so I could get an introduction to python
# gui development, and numpy. This is definetely entry level, but that is the point!
class Matrix:

    # This is the constructor for the matrix
    def __init__(self, rows, columns):
        self.rows = rows
        self.columns = columns

    # Fills the matrix with user values
    def createMatrix(self, entries):
        #print("Enter the entries in a single line (separated by space): ")
        # User input of entries in a
        # single line separated by space
        #entries = list(map(int, imput.split()))
        matrix = np.array(entries).reshape(self.rows, self.columns)
        return matrix
    # Finds the matrix P and D such that A = PDP^-1
    # Calculates eigenvalues, eigenvectors, diagonal  matrix, rounds them and takes the real part
    def PDP(self, matrix):
        eVals, eVecs = np.linalg.eig(matrix)
        invVecs = np.linalg.inv(eVecs)#the matrix P inverse
        D = np.diag(eVals)
        D = D.round(5).real
        P = eVecs.round(5).real
        Pinv = invVecs.round(5).real
        return P, D, Pinv
        #print(eVecs.round(5).real)  # The matrx P
    # Scipy implementation of LU = A factorization
    def LU(self, matrix):

        P, L, U = scipy.linalg.lu(matrix)
        L = L.round(4).real
        U = U.round(4).real
        return P, L, U
    def addMat(self, m1, m2):
        m3 = m1 + m2
        return m3
    def mul(self, m1, m2):
        m3 = np.matmul(m1, m2)
        m3 = m3.round(5).real
        return m3
    # The algorithm will return a valid QR=A decomposition, but it may be (-Q)(-R)=A, which is valid
    def QR(self, matrix):
        q, r = scipy.linalg.qr(matrix)

        q = q.round(4).real
        r = r.round(4).real
        return q, r
    def chol(self, matrix):
        L = scipy.linalg.cholesky(matrix, lower=True)
        LT = np.transpose(L)
        return L, LT





# Gets the values for the first matrix
def update():
    global answ
    Rows = testRows.get()
    Cols = testCols.get()

    answ = Entry(root, text="")
    answ.grid(column=1, row=2, columnspan = 3)

# This function handles the factorizations and their displays
def calculate():
    global answ1
    rows = testRows.get()
    columns = testCols.get()
    Mat = Matrix(rows,columns)

    read = answ.get()

    entries = list(map(int, read.split()))
    Mat1 = Mat.createMatrix(entries)

    if selection.get() == "PDP Factorization":
        P, D, Pinv = Mat.PDP(Mat1)
        display = Toplevel(root)
        result = Label(display, text="Matrix P, Diagonal matrix D, and matrix P^-1:")
        result.grid(row=0, column=0)
        resultP = Label(display, text=P)
        resultP.grid(row=1, column=0)
        resultD = Label(display, text=D)
        resultD.grid(row=1, column=1)
        resultPinv = Label(display, text=Pinv)
        resultPinv.grid(row=1, column=3)
    elif selection.get() == "LU Factorization":
        P, L, U = Mat.LU(Mat1)
        display = Toplevel(root)
        result = Label(display, text="Matrix P, Matrix L, and matrix U:")
        result.grid(row=0, column=0)
        resultP = Label(display, text=P)
        resultP.grid(row=1, column=0)
        resultL = Label(display, text=L)
        resultL.grid(row=1, column=1)
        resultU = Label(display, text=U)
        resultU.grid(row=1, column=3)
    elif selection.get() == "QR Factorization":
        Q, R = Mat.QR(Mat1)
        display = Toplevel(root)
        result = Label(display, text="Orthogonal Matrix Q, and Matrix R:")
        result.grid(row=0, column=0)
        resultQ = Label(display, text=Q)
        resultQ.grid(row=1, column=0)
        resultR = Label(display, text=R)
        resultR.grid(row=1, column=1)
    elif selection.get() == "Cholesky Factorization":
        L, LT = Mat.chol(Mat1)
        display = Toplevel(root)
        result = Label(display, text="Matrix L transpose, and Matrix L:")
        result.grid(row=0, column=0)
        resultLT = Label(display, text=LT)
        resultLT.grid(row=1, column=0)
        resultL = Label(display, text=L)
        resultL.grid(row=1, column=1)
    elif selection.get() == "Matrix Addition" or selection.get() == "Matrix Multiplication":
        display = Toplevel(root)
        Info = Label(display, text="Enter in the values of a second matrix of the same size: ")
        Info.grid(row=0, column=0)
        answ1 = Entry(display, text="")
        answ1.grid(column=0, row=2, columnspan=2)


        Done = Button(display, text="Solve", command =lambda: Solve(Mat1))
        Done.grid(row=3,column=0)

    return


# This function handles the creating of a second matrix
# and then it does the multiplication or addition afterwards
def Solve(Mat1):
    rows = testRows.get()
    columns = testCols.get()
    Mat = Matrix(rows, columns)
    read = answ1.get()

    entries1 = list(map(int, read.split()))
    Mat3 = Mat.createMatrix(entries1)
    if selection.get()== "Matrix Addition":
        Mat4 = Mat.addMat(Mat1, Mat3)
    elif selection.get() == "Matrix Multiplication":
        Mat4 = Mat.mul(Mat1,Mat3)


    display2 = Toplevel(root)
    result1 = Label(display2, text="Matrix 1, Matrix 2, and your result: ")
    result1.pack()
    result = Label(display2, text=Mat1)
    result.pack()
    result = Label(display2, text=Mat3)
    result.pack()
    result = Label(display2, text=Mat4)
    result.pack()



root = Tk()
Instructions = Label(root, text = "Enter your Rows then Columns. Then enter you matrix values Row wise: ")
Instructions.grid(row = 0, column = 0, columnspan = 4)

testRows = IntVar(root)
sizeRows = Entry(root, width=15, textvariable=testRows)
sizeRows.grid(column=2, row=1)

testCols = IntVar(root)
sizeRows = Entry(root, width=15, textvariable=testCols)
sizeRows.grid(column=3, row=1)



calc = Button(root, text="Solve", command=calculate)
calc.grid(row=2, column=0)


updatE = Button(root, text="Enter values", command=update)
updatE.grid(row=1, column=4)



Options_List = ["Matrix Multiplication", "Matrix Addition", "PDP Factorization", "LU Factorization",
                "QR Factorization",
                "Cholesky Factorization"]
selection = StringVar(root)
selection.set(Options_List[0])


# drop down menu
drop_down = OptionMenu(root, selection, *Options_List)
drop_down.grid(row=1, column=0)



root.mainloop()
