class EPWMissingDataFromFile(Exception):
    pass

class EPWFileReadFailure(Exception):
    pass

class EPWRepeatDateError(Exception):
    pass

class MaxVsAvgDistributionError(Exception):
    pass

class ExtremesIntegrationError(Exception):
    pass

class DOE2_Weather_Error(Exception):
    def __init__(self,error_message):
        self.error_message = error_message
        print(error_message)