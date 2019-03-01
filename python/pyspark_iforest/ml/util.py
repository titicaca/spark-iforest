from pyspark.ml.wrapper import JavaModel, JavaParams


def customized_from_java(java_stage):
    """
    Copied from pyspark.ml.wrapper, in order to modify the pyspark pkg name

    Given a Java object, create and return a Python wrapper of it.
    Used for ML persistence.

    Meta-algorithms such as Pipeline should override this method as a classmethod.
    """

    def __get_class(clazz):
        """
        Loads Python class from its name.
        """
        parts = clazz.split('.')
        module = ".".join(parts[:-1])
        m = __import__(module)
        for comp in parts[1:]:
            m = getattr(m, comp)
        return m

    # Change pyspark pkg name to pyspark_iforest
    stage_name = java_stage.getClass().getName().replace("org.apache.spark", "pyspark_iforest")
    # Generate a default new instance from the stage_name class.
    py_type = __get_class(stage_name)
    if issubclass(py_type, JavaParams):
        # Load information from java_stage to the instance.
        py_stage = py_type()
        py_stage._java_obj = java_stage

        # SPARK-10931: Temporary fix so that persisted models would own params from Estimator
        if issubclass(py_type, JavaModel):
            py_stage._create_params_from_java()

        py_stage._resetUid(java_stage.uid())
        py_stage._transfer_params_from_java()
    elif hasattr(py_type, "_from_java"):
        py_stage = py_type._from_java(java_stage)
    else:
        raise NotImplementedError("This Java stage cannot be loaded into Python currently: %r"
                                  % stage_name)
    return py_stage