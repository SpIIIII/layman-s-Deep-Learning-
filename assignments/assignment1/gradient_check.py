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
      
      analytic_grad_at_ix = analytic_grad[ix]
  

      # pp(x,'f(',d(x,2,ix),')- f(',d(x,-2,ix),')/',4,'   =')
      # pp(f(d(x,-2,ix)[0],'-', f(d(x,-2,ix)[0],'/',4,'   =',(f(d(x,2,ix))[0]-f(d(x,-2,ix))[0])/4)
      
      y1 = x.copy()
      y2 = x.copy()
      y1[ix]+=0.000002
      y2[ix]-=0.000002
      
      numeric_grad_at_ix = (f(y1)[0]-f(y2)[0])/0.000004
      print(f'grad_check at {ix}  num =' ,numeric_grad_at_ix,'anal =',analytic_grad_at_ix)
      # TODO compute value of numeric gradient of f to idx
      if not np.isclose(numeric_grad_at_ix, analytic_grad_at_ix, tol):
        print("Gradients are different at %s. Analytic: %2.5f, Numeric: %2.5f" % (ix, analytic_grad_at_ix, numeric_grad_at_ix))
        return False

      it.iternext()
    

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
        
def d(x,y,ix):
  in_x = np.zeros_like(x)
  in_x[ix] = x[ix]+y
  return in_x
