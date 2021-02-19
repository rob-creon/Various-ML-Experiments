"""
TODO:

-   Add discrete derivative as a model parameter, try models
    that are based on the rate of change of the price of a
    stock

-   Use batch generator or rewrite the histories array to
    be made in numpy without the list conversion

-   Migrate from AlphaVantage, use either Quantopian or
    Polygon integration

-   Maybe try using Fourier Transforms or some kind of
    averaging to derive a trend from the discrete derivative
    use this to train the model?

    -   Could also try derivative over N days

-   Try generating 30 day forecasts

-   Refactor Model class so that it is a abstract class that
    allows its derived classes to either be ML algorithms or
    non-ML algorithms, or hybrid

        -   Partially implemented, next step is to add a deep
            learning base class and a shallow learning base
            class

-   Try correlative stocks such as competitors and related
    commodities

-   Fetch commodities?

-   Refactor datasets so that they can be passed as human
    readable format to a model and then the model can turn
    it into model params if it needs to, allows for more
    complex non-deep learning models

-   Analysis tools for identifying weaknesses and strengths
    of a model is it consistently too high/low, or does it
    just lean towards the average? etc.

-   Add EMA and SMA to StockData, probably fetch from
    AlphaVantage (likely better than my own)

-   Smooth corners off of EMA and SMA, handle cases for
    first i in range points, use older history data when
    possible

"""