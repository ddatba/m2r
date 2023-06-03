import numpy as np
import matplotlib.pyplot as plt


class Dual:
    
    def __init__(self, real, dual):
        '''
        real: real number
        dual: dict (key=name_index and value=value)
        '''
        self.real = real
        self.dual = dual
        
    def __add__(self, argument):
        if isinstance(argument, Dual):
            real = self.real + argument.real
            dual = {}
            for key in self.dual:
                dual[key] = self.dual[key]
            for key in argument.dual:
                if key in dual:
                    dual[key] += argument.dual[key]
                else:
                    dual[key] = argument.dual[key]    
            return Dual(real, dual)
        else:
            return Dual(self.real + argument, self.dual)
        
    __radd__ = __add__
    
    def __sub__(self, argument):
        if isinstance(argument, Dual):
            real = self.real - argument.real
            dual = {}
            for key in self.dual:
                dual[key] = self.dual[key]
            for key in argument.dual:
                if key in dual:
                    dual[key] -= argument.dual[key]
                else:
                    dual[key] = -argument.dual[key]    
            return Dual(real, dual)
        else:
            return Dual(self.real - argument, self.dual)
        
    def __rsub__(self, argument):
            return -self + argument
    
    def __mul__(self, argument):
        if isinstance(argument, Dual):
            real = self.real * argument.real
            dual = {}
            for key in self.dual:
                dual[key] = self.dual[key] * argument.real
            for key in argument.dual:
                if key in dual:
                    dual[key] += argument.dual[key] * self.real
                else:
                    dual[key] = argument.dual[key] * self.real
            return Dual(real, dual)
        else:
            dual = {}
            for key in self.dual:
                dual[key] = self.dual[key] * argument
            return Dual(self.real * argument, dual)
        
    __rmul__ = __mul__
  
    def __truediv__(self,argument):
        if isinstance(argument, Dual):
            x = argument.real
            new_arg = self.div_neg(argument)
            num = Dual(self.real, self.dual)
            num_modified = num*new_arg
            dual = {}
            for key in num_modified.dual:
                dual[key] = num_modified.dual[key] / (x*x)
            return Dual(num_modified.real / (x*x), dual)
        else:
            dual = {}
            for key in self.dual:
                dual[key] = self.dual[key] / argument
            return Dual(self.real / argument, dual)

    def __rtruediv__(self,argument):
        x = self.real
        den = Dual(self.real, self.dual)
        new_arg = self.div_neg(den)
        num_modified = argument*new_arg
        dual = {}
        for key in num_modified.dual:
            dual[key] = num_modified.dual[key] / (x*x)
        return Dual(num_modified.real / (x*x), dual)
        
    def __pow__(self, power):
        a = self.real
        dual = {}
        for key in self.dual:
            dual[key] = power*self.dual[key]*(a**(power-1))
        return Dual(a**power,dual)
    
    def __neg__(self):
        dual = {}
        for key in self.dual:
            dual[key] = self.dual[key]*(-1)
        return Dual(-self.real,dual)

    def div_neg(self, argument):
        dual = {}
        for key in argument.dual:
            dual[key] = argument.dual[key]*(-1)
        return Dual(argument.real,dual)
    
    def __str__(self):
        s = 'f = ' + str(round(self.real,6)) + '\n'
        for key in self.dual:
            s += 'f' + key + ' = ' + str(round(self.dual[key],6)) + '\n'
        return s

def log_d(dual_number):
    if isinstance(dual_number, Dual):
        dual = {}
        a = dual_number.real
        sa = np.log(a)
        for key in dual_number.dual:
            dual[key] = dual_number.dual[key]/a
        return Dual(sa, dual)
    else:
        return np.log(dual_number)
    
def exp_d(dual_number):
    if isinstance(dual_number, Dual):
        dual = {}
        a = dual_number.real
        sa = np.exp(a)
        for key in dual_number.dual:
            dual[key] = dual_number.dual[key]*sa
        return Dual(sa, dual)
    else:
        return np.exp(dual_number)

def sin_d(dual_number):
    if isinstance(dual_number, Dual):
        dual = {}
        a = dual_number.real
        sa = np.sin(a)
        for key in dual_number.dual:
            dual[key] = dual_number.dual[key]*np.cos(a)
        return Dual(sa, dual)
    else:
        return np.sin(dual_number)

def cos_d(dual_number):
    if isinstance(dual_number, Dual):
        dual = {}
        a = dual_number.real
        sa = np.cos(a)
        for key in dual_number.dual:
            dual[key] = -np.sin(a)*dual_number.dual[key]
        return Dual(sa, dual)
    else:
        return np.cos(dual_number)
    
def sigmoid_d(dual_number):
    if isinstance(dual_number, Dual):
        dual = {}
        a = dual_number.real
        sa = 1 / (1 + np.exp(-a))
        for key in dual_number.dual:
            dual[key] = dual_number.dual[key]*sa*(1-sa)
        return Dual(sa, dual)
    else:
        return 1 / (1 + np.exp(-dual_number))

def C(p, eps=1e-10):  #C-value in wA
    if np.all(isinstance(p, Dual)):
        q = p.real
    else:
        q = p
    if np.all(np.abs(q)<1):
        n=2
        c = (1-p**2)**2
        while p**(2*n)>=eps:
            c *= (1-p**(2*n))**2
            n +=1
        return c
    else:
        raise Exception('invalid p')

def P(z,p, eps=1e-10):  #P-value in wA
    zval = z
    if np.all(isinstance(z, Dual)):
        Z = zval.real
    else:
        Z = zval
    if np.all(isinstance(p, Dual)):
        q = p.real
    else:
        q = p
    if np.any(Z==0):
        if np.all(isinstance(Z,np.ndarray)):
            raise Exception('z=0 unsupported for array')  
        elif q==0:
            return 1 - z
        else:
            raise Exception('z=0 is invalid') 
    if np.all(np.abs(q)<1):
        n = 2
        a = (1-zval*p**2)*(1-(p**2)/zval)
        while np.any(np.abs((Z+1/Z)*q**(2*n)-q**(4*n))>=eps):
            a *= (1-zval*p**(2*n))*(1-(p**(2*n))/zval)
            n +=1
        return (1-z)*a
    else:
        raise Exception('invalid p')
    
def wA(z,a,p, eps=1e-10):  #prime function of annulus
    return -a*P(z/a,p,eps**2)/C(p, eps**2)

def diff1(f1, val):
    x = f1(Dual(real=val, dual={'1': 1}))
    return x.dual['1']

def diff2(f1, f2, val, method='l'):  #Jacobean matrix, set method='a' if val is an array
    if method == 'l':
        n = len(val)
        x = []
        for k in range(n):
            x += [Dual(real=val[k], dual={f'x{k}': 1})]
        return np.array([[f1(*x).dual[f'x{k}'] for k in range(n)],
                         [f2(*x).dual[f'x{k}'] for k in range(n)]])
    elif method == 'a':
        m, n = np.shape(val)
        x = []
        for k in range(n):
            x += [Dual(real=0, dual={f'x{k}': 1})]
        xval = val + np.tile(x,(m,1))
        return np.array([[[f1(*xval[i]).dual[f'x{k}'] for k in range(n)], [f2(*xval[i]).dual[f'x{k}'] for k in range(n)]] for i in range(m)])
    else:
        raise Exception('invalid method')

def newt1(f1, F, val, eps=1e-10, n=50): 
    '''
        One-dimension Newtons method for solving f1(x)=F. 

        Parameters: 
            f1 (function): Function of equation 
            F (float/array): Desired output(s) of f1 
            val (float/array): Point(s) of initial guess 
            *eps (float): Desired error bound 

        Return:  
            estimate (float/array): Estimate of solution(s) with the same type as val 
    '''
    #initialisation
    m = n
    df = diff1(f1, val)
    f = F-f1(val)
    delta = f/df
    nval = val
    #loop
    while np.any(abs(delta)>=err) and m>0:
        nval = nval + delta
        df = diff1(f1, nval)
        f = F-f1(nval)
        delta = f/df
        m -= 1
    if m>0:
        return nval + delta
    else: 
        raise Exception('max iteration reached')    

def newt2(f1, f2, F, val, eps=1e-10, n=100, method='l'):  
    '''
        Two-dimension Newtons method for solving f1(x)=F1 and f2(x)=F2. 

        Parameters: 
            f1 (function): First function 
            f2 (function): Second function 
            F (list/array): Desired output(s) of f1, f2 in form of [F1, F2]     
            val (list/array): Point(s) of initial guess in form of [val1, val2] 
            *eps (float): Desired error bound 
            *method (string): if method = 'a' then F and val must be arrays 

        Return:  
            estimate (array): Estimate of solution(s) 
    '''
    m = n
    if method == 'l':
        #initialisation
        df = diff2(f1, f2, val)
        f = np.array([-f1(*val), -f2(*val)])+np.array(F)
        delta = np.linalg.solve(df, f)
        nval = np.array(val)
        #loop
        while np.any(abs(delta)>=err) and m>0:
            nval = nval + delta
            df = diff2(f1, f2, nval)
            f = np.array([-f1(*nval), -f2(*nval)])+np.array(F)
            delta = np.linalg.solve(df, f)
            m -= 1
        if m>0:
            return nval + delta
        else: 
            raise Exception('max iteration reached')
    elif method == 'a':
        #initialisation
        df = diff2(f1, f2, val, method = 'a')
        f = np.array([-f1(*val.T), -f2(*val.T)]).T + np.array(F)
        delta = np.linalg.solve(df, f)
        nval = np.array(val)
        #loop
        while np.any(abs(delta)>=err) and m>0:
            nval = nval + delta
            df = diff2(f1, f2, nval, method = 'a')
            f = np.array([-f1(*nval.T), -f2(*nval.T)]).T+np.array(F)
            delta = np.linalg.solve(df, f)
            m -= 1
        if m>0:
            return nval + delta
        else: 
            raise Exception('max iteration reached')
    else:
        raise Exception('invalid method')  

def cont(f1, f2, d2, val, d1=1, F0=[0,1], n=10, m=10): 
    '''
        Continuation method for solving f1(x)=d1 and f2(x)=d2. 

        Parameters: 
            f1 (function): First function 
            f2 (function): Second function 
            d2 (float): Desired output of f2 
            val (list): Point of initial guess in form of [val1, val2] 
            *d1 (float): Desired output of f1 
            *F0 (list): Output of initial guess in form of [f1(val), f2(val)] 
            *n (int): Number of iterations to reach d1 
            *m (int): Number of iterations to reach d2 

        Return:  
        estimate (float): Estimate of solution 
    '''
    s = np.linspace(((n-1)*F0[0]+d1)/n,d1,n)
    x = val
    for i in s:
        F = [i, F0[1]]
        x = newt2(f1, f2, F, x)

    t = np.linspace((F0[1]+d2)/m, d, m)
    for i in t:
        F = [d1, i]
        x = newt2(f1, f2, F, x)

    return x

def ceffn(f1, f2, fd = lambda D : [D,D+1], fval = lambda D : [D, 1/2+0*D], mind=1, maxd=10000, m=1000, method='a'):
    '''
        Generates graph of field ceff over d using newt2 to solve f1(R,p)=fd1(d) and f2(R,p)=fd2(d) 

            Parameter: 
                f1 (function): First function 
            f2 (function): Second function 
            *fd (function): Function of d in form of [fd1, fd2] 
            *fval (function): Function of d for initial guess in form of [fval1, fval2] 
            *mind (float): Minimum d of graph 
            *maxd (float): Maximum d of graph 
            *m (int): Number of steps between mind and maxd 
            *method (string): if method = 'l' then list would be used instead of array 
        **note: fd1, fd2, fval1, fval2 must all contain variable d i.e. 0*d 

        Return: 
            graph (plot): Plot of ceff over d 
    '''
    if method == 'l':
        d = np.linspace(mind,maxd,m)
        pval = []
        for i in d:
            val = fval(i)
            F = fd(i)
            pval.append(-2*np.pi/np.log(newt2(f1, f2, F, val)[1]))
        plt.plot(d,pval)
    elif method == 'a':
        d = np.linspace(mind,maxd,m)
        val = np.array(fval(d)).T
        F = np.array(fd(d)).T
        plt.plot(d,-2*np.pi/np.log(newt2(f1, f2, F, val, method='a').T[1]))
    else:
        raise Exception('invalid method')
    
def ceffc(f1, f2, val, d1=1, F0=[0,1], n=2,m=2,mind=1, maxd=100, k=100):
    '''
        Generates graph of field ceff over d using cont2 to solve f1(R,p)=d1 and f2(R,p)=d 

        Parameter: 
            f1 (function): First function 
            f2 (function): Second function 
            val (list): Point of initial guess in form of [val1, val2] 
            *d1 (float): Desired output of f1 
            *F0 (list): Output of initial guess in form of [f1(val), f2(val)] 
            *n (int): Number of iterations to reach d1 
            *m (int): Number of iterations to in between the d values 
            *mind (float): Minimum d of graph 
            *maxd (float): Maximum d of graph 
            *k (int): Number of steps between mind and maxd 

        Return: 
            graph (plot): Plot of ceff over d 
    '''
    dval = np.linspace(mind,maxd,k)
    s = np.linspace(((n-1)*F0[0]+d1)/n,d1,n)
    x = val
    pval = []
    for i in s:
        F = [i, F0[1]]
        x = newt2(f1, f2, F, x)
    if mind==F0[1]:
        pval += [x[1]]
    else:
        t = np.linspace(((m-1)*F0[1]+mind)/m, mind, m)
        for j in t:
            F = [1, j]
            x = newt2(f1, f2, F, x)  
        pval += [x[1]]
    
    for i in range(k-1):
        t = np.linspace(((m-1)*dval[i]+dval[i+1])/m, dval[i+1], m)
        for j in t:
            F = [1, j]
            x = newt2(f1, f2, F, x)  
        pval += [x[1]]
    plt.plot(dval, -2*np.pi/np.log(np.array(pval)))

def hplot(h, p, n=30, m = 500, xbound = [-2,2], ybound = [-2,2], figsize= 10, ax=True, shift=False, theta=np.pi/2):
    '''
        Generates plot of feild lines from h

        Parameters:
            h (function): h of desired graph
            p (float): p-value of annulus
            *n (int): Number of field lines
            *xbound (list): Boundary of x in form of [xmin, xmax]
            *ybound (list): Boundary of x in form of [xmin, xmax]
            *figsize (float): Desired size of plot
            *ax (bool): Hides axis if False
            *shift (bool): Shifts the branch cut of log if True
            *theta (float): Argument of new branch cut
    
        Return 
            graph (plot): Plot of field lines
    '''

    x = np.linspace(xbound[0],xbound[1],m )
    y = np.linspace(ybound[0],ybound[1],m)
    xval, yval = np.meshgrid(x,y)
    z = xval + yval * 1j
    f = h(z)
    Re = f.real
    Im = f.imag
    Im = Im.astype(float)

    Im = np.where(Re<=1, Im, np.nan)
    Im = np.where(Re>=0, Im, np.nan)

    if shift:
        if -np.pi<theta<np.pi:
            if theta>0:
                Im = np.where(Im>=theta/s, Im-2*np.pi/np.abs(np.log(p)), Im)
            else:
                Im = np.where(Im<=theta/s, Im+2*np.pi/np.abs(np.log(p)), Im)
        else:
            raise Exception('invalid theta')

    imax = np.nanmax(Im)
    imin = np.nanmin(Im)
    b = (imax-imin)/(2*n)
    
    Im = np.where(Im<imax-b/4, Im, np.nan)
    Im = np.where(Im>imin+b/4, Im, np.nan)

    plt.figure(figsize=[figsize,figsize])
    plt.contour(xval, yval, Im, np.linspace(imin+b,imax-b,n), colors='black', linestyles='solid')
    plt.contour(xval, yval, Re, [0,1], colors=['blue', 'red'])
    plt.axis('scaled')
    plt.axis(xbound+ybound)
    if not ax:
        plt.axis('off')

def wplot(w, p, n=30, m=0, xbound = [-2,2], ybound = [-2,2], figsize=10, kn=1000, km=1000, ax=True):
    '''
        Generates plot of field lines from transformation of annulus.

            Parameter:
                w (function): Transformation function
                p (float): p-value of annulus
                *n (int): Number of electric field lines
                *m (int): Number of potential lines other than the two electrode lines
                *xbound (list): Boundary of x in form of [xmin, xmax]
                *ybound (list): Boundary of x in form of [xmin, xmax]
                *figsize (float): Desired size of the plot
                *kn (int): Number of samples used for field lines (for smoothness)
                *km (int): Number of samples used for potential lines (for smoothness)
                *ax (bool): Display axis if True
    
            Return:
                graph (plot): Plot of feild lines
    '''
    t = np.linspace(0,2*np.pi,n, endpoint=False)
    r = np.linspace(p,1,kn)
    plt.figure(figsize=[figsize,figsize])
    for i in t:
        a = r*np.cos(i)+r*np.sin(i)*1j
        k=w(a)
        plt.plot(k.real,k.imag, color='black')

    t2 = np.linspace(0,2*np.pi,km)
    r2 = np.linspace(p,1,m+2)
    color = ['red']+m*['black']+['blue']
    for i in range(m+2):
        a = r2[i]*np.cos(t2)+r2[i]*np.sin(t2)*1j
        k=w(a)
        plt.plot(k.real,k.imag, color=color[i])

    plt.axis('scaled')
    plt.axis(xbound+ybound)
    if not ax:
        plt.axis('off')
