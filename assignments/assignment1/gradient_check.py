import numpy as np


def check_gradient(f, x, delta=1e-5, tol = 1e-4):
    '''
    Checks the implementation of analytical gradient by comparing
    it to numerical gradient using two-point formula

    Arguments:
      f: function that receives x and computes value and gradient
      x: np array, initial point where gradient is checked
      delta: step to compute numerical gradient
      tol: tolerance for comparing numerical and analytical gradient

    Return:
      bool indicating whether gradients match or not
    '''
    
    assert isinstance(x, np.ndarray)
    assert x.dtype == np.float
    
    orig_x = x.copy()
    fx, analytic_grad = f(x)
    assert np.all(np.isclose(orig_x, x, tol)), "Functions shouldn't modify input variables"

    assert analytic_grad.shape == x.shape
    analytic_grad = analytic_grad.copy()

    # We will go through every dimension of x and compute numeric
    # derivative for it
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    
    while not it.finished:
      
      ix = it.multi_index
      zeros = np.zeros_like(x)
      zeros[ix]=x[ix]
      analytic_grad_at_ix = analytic_grad[ix]
      
      pp('f(',(x+2),')- f(',(x-2),')/',4,'   =')
      pp(f(x+2)[0],'-', f(x-2)[0],'/',4,'   =',(f(x+2)[0]-f(x-2)[0])/4)
      
      numeric_grad_at_ix = (f(x+2)[0]-f(x-2)[0])/4

      # TODO compute value of numeric gradient of f to idx
      if not np.isclose(numeric_grad_at_ix, analytic_grad_at_ix, tol):
        print("Gradients are different at %s. Analytic: %2.5f, Numeric: %2.5f" % (ix, analytic_grad_at_ix, numeric_grad_at_ix))
        return False

      it.iternext()
    
    if not analytic_grad == (f(x+2)[0]-f(x-2)[0])/4:
      return False

    print("Gradient check passed!")
    return True

        
def fmt_items(lines,max_lines=0):
    max_width=max([len(line)for line in lines])
    empty =' '*max_width
    lines = [line.ljust(max_width)for line in lines]
    lines += [empty]*(max_lines - len(lines))
    return lines
def pp (*list):
    lines = [ str(item).split('\n') for item in list]
    max_lines=max([len(item)for  item in lines])
    lines = [fmt_items(item,max_lines=max_lines)for item in lines]
    lines_t= np.array(lines).T
    print('\n'.join([' '.join(line) for  line in lines_t]))
        
