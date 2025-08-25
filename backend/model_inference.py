from abc import ABC,abstractmethod
import inspect


"""Base class for all the predictors.
   Since we cant create an instance of abstract class, we assume parameters (attributes) are initialzied by the child model class. """

class BasePredictor(ABC):
   
    @abstractmethod
    def fit(self, X, y):
        """Fit predictor.
        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        y : ndarray of shape (n_samples,)
            The input target value. 


        Returns
        -------
        self : object
            Fitted estimator.

        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
        """
        pass

    
    # copied from sklearn 
    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [
            p
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    "scikit-learn estimators should always "
                    "specify their parameters in the signature"
                    " of their __init__ (no varargs)."
                    " %s with constructor %s doesn't "
                    " follow this convention." % (cls, init_signature)
                )
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    #copied from sklearn
    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, "get_params"):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out

    
    def reset(self):
       """A method for reseting the predictor"""   
       new = self.__class__(**self.get_params())
       return new

    
    def load_params(self, params):
      """A method to load model configurations.
      
      Parameters
      -----------
      params : dict of parameters
  
      Returns
      ---------
      A new model instance with the new parameters.  

      """

      self = self.__class__(**params)
      print("params loaded")
      
      return self
      