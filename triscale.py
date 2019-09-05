"""
TriScale modules
- network_profiling
- experimental_design
- preprocessing
- analysis
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from helpers import convergence_test, ThompsonCI, stationarity_test, min_number_samples, repeatability_test
from triplots import theil_plot, autocorr_plot, ThompsonCI_plot

# ----------------------------------------------------------------------------------------------------------------------------
# NETWORK PROFILING MODULE
# ----------------------------------------------------------------------------------------------------------------------------

def network_profiling(df, link_quality_bounds, link_quality_name=None,
             print_output=True):
    '''
    Takes a data frame as input containing two columns:
    - timestamps (any format?)
    - average link quality values
    '''

    todo = ''
    todo += '# ---------------------------------------------------------------- \n'
    todo += '# TODO network_profiling \n'
    todo += '# ---------------------------------------------------------------- \n'
    todo += '- verbose display (clearly) advising people what they should do\n'
    todo += 'i.e., how long should be the time span of an experiment\n'
    todo += '- write a doctring\n'
    todo += '- write output to file\n'
    todo += '- input check\n'
    todo += '- iterate through the list of inputs to see if we have all we need\n'
    todo += '- handle the intermediary outputs\n'
    todo += '- format the final output\n'
    todo += '- add a test on the minimal number of samples required to be left in the autocorellation\n'
    todo += '# ---------------------------------------------------------------- \n'
    print('%s' % todo)

    ## other inputs I need
    confidence_convergence = 95
    confidence_repeatability = 95
    tolerance_convergence = 1
    network_name = 'FlockLab - DPP-cc430'

    profiling_output = ''
    profiling_output += '# ---------------------------------------------------------------- \n'
    profiling_output += '# TriScale report - Network profiling\n'
    profiling_output += '# ---------------------------------------------------------------- \n'
    profiling_output += '\nNetwork \t%s\n' % (network_name)


    ##
    # Checking the inputs
    ##

    ##
    # Data must be a dataframe with (at least) two columns
    # - link_quality
    # - date_time (can also be the index)
    if 'link_quality' not in df.columns:
        raise ValueError("Wrong input. The 'data' DataFrame must contain a 'link_quality' column.")

    if ('date_time' not in df.columns) and (not isinstance(df.index, pd.DatetimeIndex)):
        raise ValueError("Wrong input. The 'data' DataFrame must contain a 'date_time' column or have DatetimeIndex type.")

    # Parse dates
    if 'date_time' in df.columns:
        df['date_time'] = pd.to_datetime(df['date_time'], utc=True)
        df.set_index('date_time')
    # Make sure the DataFrame is sorted
    df.sort_index(inplace=True)

    profiling_output += '\nProfiling time span\n'
    profiling_output += 'from \t\t%s\n' % df.index[0]
    profiling_output += 'to \t\t%s\n' % df.index[-1]
    profiling_output += '\nProfiling granularity\n'
    profiling_output += '\t\t%s\n' % (df.index[1] - df.index[0])
    profiling_output += '\n# ---------------------------------------------------------------- \n'

    ##
    # Convergence test
    ##

    results = convergence_test(
                    df.index,
                    df.link_quality.values,
                    link_quality_bounds,
                    confidence_convergence,
                    tolerance_convergence,
                    # y_label=link_quality_name,
                    # plot=True
                    )
    # Plot the time series
    plot_out_name=None
    default_layout={'title' : ('Regression on %s' % link_quality_name),
                    'xaxis' : {'title':None},
                    'yaxis' : {'title':link_quality_name}}
    figure = theil_plot(    df.link_quality.values,
                            x=df.index,
                            convergence_data=results,
                            layout=default_layout,
                            out_name=plot_out_name)
    figure.show()


    ##
    # Stationarity test
    ##

    # Replace missing samples with the series median
    # -> We need continuous data for autocorrelation
    data = df.link_quality.values
    data[np.isnan(data)] = np.nanmedian(data)

    stationary = stationarity_test(data)
    if stationary:
        profiling_output += '\nNetwork link quality appears I.I.D. (95%% confidence)\n'
    else:
        profiling_output += '\nNetwork link quality does NOT appears I.D.D. !\nSearching for a suitable time interval...\n\n'

    # Plot the autocorrelation
    figure = autocorr_plot(data)
    figure.show()

    # Search for a suitable test window
    window_size = 1
    while not stationary:
        window_size += 1
        data_subsampled = [np.nanmean(np.array(data[i:i+window_size])) for i in np.arange(0, len(data), window_size)]
        stationary = stationarity_test(data_subsampled)
        profiling_output += 'Window size: %g\tStationary: %i\n' % (window_size, stationary)
#         plot_autocorr(data_subsampled)

    # Compute the corresponding time span
    time_span = df.index[window_size] - df.index[0]

    profiling_output += '\n\nWith a confidence of 95%\n'
    profiling_output += 'network link quality appears stationary over a \n'
    profiling_output += 'time span of'
    profiling_output += '\t%s\n' % (time_span)
    profiling_output += '\n# ---------------------------------------------------------------- \n'

    ## DEPRECATED
#     ##
#     # Network variability
#     ##
#     network_variability = (repeatability_test(data, confidence_repeatability=confidence_repeatability))
# #     print(network_variability*100)
#
#     profiling_output += '\nNetwork variability (%g%% confidence level)\n' %confidence_repeatability
#     profiling_output += '\t\t%0.2f %%\n' % (network_variability*100)
#     profiling_output += '\n# ---------------------------------------------------------------- \n'

    if print_output:
        print(profiling_output)

    return

# ----------------------------------------------------------------------------------------------------------------------------
# EXPERIMENT DESIGN MODULE
# ----------------------------------------------------------------------------------------------------------------------------

def experiment_design(percentile,
                      confidence,
                      nb_samples=0,
                      robustness=0):
    todo = ''
    todo += '# ---------------------------------------------------------------- \n'
    todo += '# TODO \n'
    todo += '# ---------------------------------------------------------------- \n'
    todo += '- account for the double-sided/single-sided case\n'
    todo += '- verbose display (clearly) advising people what they should do\n'
    todo += 'i.e., how many samples in a data serie must be collected\n'
    todo += '- write a doctring\n'
    todo += '- write output to file (?) I don\'t think that\'s necessary here...\n'
    todo += '- iterate through the list of inputs to see if we have all we need\n'
    todo += '- adapt to make ci() return a dictionaty\n'
    todo += '- rename ci() -> ci_thompson? thompson_ci? \n'
    todo += '# ---------------------------------------------------------------- \n'
    print('%s' % todo)

    ##
    # Checking the inputs
    ##

    if confidence >= 100 or confidence <= 0:
        raise ValueError("Invalid confidence: "+repr(confidence)+". Provide a real number strictly between 0 and 100.")
    if percentile >= 100 or percentile <= 0:
        raise ValueError("Invalid percentile: "+repr(percentile)+". Provide a real number strictly between 0 and 100.")
    if not isinstance(robustness, int):
        raise ValueError("Invalid robustness: "+repr(robustness)+". Provide a positive integer.")
    if robustness < 0:
        raise ValueError("Invalid robustness: "+repr(robustness)+". Provide a positive integer.")
    if not isinstance(nb_samples, int):
        raise ValueError("Invalid nb_samples: "+repr(nb_samples)+". Provide a positive integer.")
    if nb_samples < 0:
        raise ValueError("Invalid nb_samples: "+repr(nb_samples)+". Provide a positive integer.")

    # Work with lower-percentiles
    if percentile > 50:
        wk_perc = 100-percentile
    else:
        wk_perc = percentile

    # If the number of samples is not specified, compute the minimal number
    # necessary to estimate the given percentile ant the given confidence
    # (possibly with a given number of data sample to omit = robustness)
    if not nb_samples:
        N_single,N_double = min_number_samples(wk_perc,confidence,robustness)
        print(N_single,N_double)

    # Otherwise, check whether the given number of samples is sufficent
    else:
        dummy_data = np.ones(nb_samples)
        result = ci([dummy_data], wk_perc, confidence)
        print(result)
        if result is not None and result[0][0] <= robustness:
            print("%g samples are not enough to exclude %g samples from the CI." % (nb_samples, robustness))

    return


# ----------------------------------------------------------------------------------------------------------------------------
# ANALYSIS MODULE
# ----------------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------------
# Preprocessing
#
def analysis_preprocessing(data_file,
                           metric,
                           plot=False,
                           plot_out_name=None,
                           convergence=None,
                           verbose=False):

    todo = ''
    todo += '# ---------------------------------------------------------------- \n'
    todo += '# TODO \n'
    todo += '# ---------------------------------------------------------------- \n'
    todo += '- modify the ploting: it should plot regardless of whether the convergence test is run or not\n'
    todo += '-> Take the theil_plot function out of the convergence test and put in here directly. Adapt to show the regression and bounds only when convergence is expected\n'
    todo += '\n'
    todo += '- write the doctring\n'
    todo += '- modif convergence_test() to output a dictionary\n'
    todo += '- check for crazy values in the input dictionaries\n'
    todo += '- polish the plot: display tol and conf in the legend\n'
    todo += '# ---------------------------------------------------------------- \n'
    if verbose:
        print('%s' % todo)

    ##
    # Checking the inputs
    ##

    # Data file
    if not isinstance(data_file, str):
        raise ValueError("Wrong input type. Expect a string, got "+repr(data_file)+".")
    try:
        df = pd.read_csv(data_file, delimiter=',', names=['x', 'y'], header=0)
    except FileNotFoundError:
        print(repr(data_file) + " not found")

    if len(df.index) == 0:
        print("Input file is empty (%s)" % data_file )
        return False, np.nan


    # Metric
    if 'bounds' not in metric:
        metric['bounds'] = [df.y.min(), df.y.max()]

    if 'name' not in metric:
        metric['name'] = None
        metric_label = ''
    else:
        metric_label = metric['name']
        if 'unit' in metric:
            metric_label += ' [' + metric['unit'] + ']'

    # Convergence
    if convergence is not None and convergence['expected'] == True:

        # Convergence test should run
        run_convergence_test = True

        # Check the confidence and tolerance values
        if 'confidence' not in convergence:
            # Default to 95% confidence
            convergence['confidence'] = 95
        if 'tolerance' not in convergence:
            # Default to 1% tolerance
            convergence['tolerance'] = 1
    else:
        run_convergence_test = False

    ##
    # Convergence test
    ##
    if run_convergence_test:
        results = convergence_test(df.x.values,
                                   df.y.values,
                                   metric['bounds'],
                                   convergence['confidence'],
                                   convergence['tolerance'],
                                   verbose=verbose)

        if results[0]:
            has_converged = True
        else:
            has_converged = False

        # Produce the output string
        if verbose:
            if has_converged:
                flag_convergence1 = '[ PASSING ]'
                flag_convergence2 = ''
                preprocessing_warning = '\n'
            else:
                flag_convergence1 = '[ FAILED ]'
                flag_convergence2 = 'NOT '
                preprocessing_warning = '\n[ WARNING ] These data should not be used to estimate \nthe long-term performance of the system under test!\n'

            preprocessing_output = ''
            preprocessing_output += '%s\n' % flag_convergence1
            preprocessing_output += 'With a confidence level of \t%g%%\n' % (convergence['confidence'])
            preprocessing_output += 'given a tolerance of \t\t%g%%\n' % (convergence['tolerance'])
            preprocessing_output += 'Run has %sconverged.\n' % flag_convergence2
            preprocessing_output += '%s' % preprocessing_warning

            print(preprocessing_output)
    else:
        results = None

    ##
    # Plot
    ##
    if plot:
        default_layout={'title' : ('Regression on original %s' % metric_label),
                        'xaxis' : {'title':None},
                        'yaxis' : {'title':metric_label}}
        figure = theil_plot(    df.y.values,
                                x=df.x.values,
                                convergence_data=results,
                                layout=default_layout,
                                out_name=plot_out_name)
        figure.show()

    ##
    # Return the run's measure
    ##
    # Test failed
    if run_convergence_test:
        if not has_converged:
            return False, np.nan
    # Test passed or no test
    return True, np.percentile(df.y.values, metric['measure'], interpolation='midpoint')




# ----------------------------------------------------------------------------------------------------------------------------
# KPI computation
#
def analysis_kpi(data,
                 KPI,
                 to_plot=None,
                 verbose=False):
    '''

    '''


    todo = ''
    todo += '# ---------------------------------------------------------------- \n'
    todo += '# TODO analysis_kpi \n'
    todo += '# ---------------------------------------------------------------- \n'
    todo += '- write the doctring\n'
    todo += '- work on the "default layout" for the plots (axis, title, written info)\n'
    todo += '# ---------------------------------------------------------------- \n'
    if verbose:
        print('%s' % todo)


    ##
    # Input checks
    ##

    # Data can be either a csv file or a list
    # for now we write the case with the list..


    # For now, we assume the inputs are correct...
    output_log = ''

    ##
    # Stationarity test
    ##
    stationary = stationarity_test(data)

    if stationary:
        output_log += ('Data appears i.i.d. (95%% confidence)\n')
    else:
        output_log += ('Data appears NOT I.D.D. !\n')
        output_log += ('Analysis continues but results are not trustworthy...')

    if verbose:
        print(output_log)

    ##
    # Compute the KPI
    ##
    KPI_bounds = ThompsonCI(len(data),
                           KPI['percentile'],
                           KPI['confidence'],
                           KPI['class'],
                           verbose)
    ##
    # Plots
    ##
    if to_plot is not None:
        if 'autocorr' in to_plot:
            autocorr_plot( data )

        layout = go.Layout(
            title='Key Performance Indicator',
            width=500)
        if not np.isnan(KPI_bounds[0]):
            if 'horizontal' in to_plot:
                ThompsonCI_plot( data, KPI_bounds, KPI['side'], 'horizontal', layout)
            if 'vertical' in to_plot:
                ThompsonCI_plot( data, KPI_bounds, KPI['side'], 'vertical', layout, out_name=None)


    return stationary, KPI_bounds

def analysis_repeatability(data,
                           confidence,
                           to_plot=None,
                           verbose=False):

    todo = ''
    todo += '# ---------------------------------------------------------------- \n'
    todo += '# TODO analysis_repeatability \n'
    todo += '# ---------------------------------------------------------------- \n'
    todo += '- write the doctring\n'
    todo += '- polish the plots\n'
    todo += '# ---------------------------------------------------------------- \n'
    print('%s' % todo)

    ##
    # Input checks
    ##

    # Data can be either a csv file or a list
    # for now we write the case with the list..


    # For now, we assume the inputs are correct...
    output_log = ''

    ##
    # Stationarity test
    ##
    stationary = stationarity_test(data)

    if stationary:
        output_log += ('Data appears i.i.d. (95%% confidence)\n')
    else:
        output_log += ('Data appears NOT I.D.D. !\n')
        output_log += ('Analysis continues but results are not trustworthy...')

    if verbose:
        print(output_log)

    ##
    # Compute the repeatability bounds
    ##
    repeatability_bounds = ThompsonCI(len(data),
                                      50,
                                      confidence,
                                      'two-sided',
                                      verbose)

    ##
    # Plots
    ##
    if to_plot is not None:
        if 'autocorr' in to_plot:
            plot_autocorr( data )

        layout = go.Layout(
            title='Repeatability'
        )

        if ('1D' in to_plot or 'cumdist' in to_plot or 'hist' in to_plot) and not np.isnan(repeatability_bounds[0]):
            plot_CI( data, repeatability_bounds, 'two-sided', to_plot, layout)

    ##
    # Compute the repeatability score
    ##
    data.sort()
    repeatability_bounds_values = [ data[repeatability_bounds[k]] for k in [0,1] ]
    repeatability_mean_value = (repeatability_bounds_values[0]+repeatability_bounds_values[1])/2
    if repeatability_bounds[0] != np.nan:
        repeatability_score = data[repeatability_bounds[1]] - data[repeatability_bounds[0]]
    else:
        repeatability_score = np.nan


    return stationary, repeatability_mean_value, repeatability_bounds_values, repeatability_score

def analysis_report(data, meta_data, output_file_name=None):
    '''
    Prepare and output the TriScale performance report based on the profided data, which may be
    - a list (of list) of file names, which contain the raw data
    - a list (of list) of values, which are the metric values for each run and series
    - a list of values, which are the KPI for each series
    '''

    todo = ''
    todo += '# ---------------------------------------------------------------- \n'
    todo += '# TODO analysis_report \n'
    todo += '# ---------------------------------------------------------------- \n'
    todo += '- finish the doctring\n'
    todo += '- check input files\n'
    todo += '# ---------------------------------------------------------------- \n'
    print('%s' % todo)

    ##
    # Input checks
    ##

    # format of meta_data? probably a dictionary...
    metric = {'name': 'Throuput',
          'unit': 'MBit/s',
          'measure': 95,
          'bounds': [0,120]}
    convergence = {'expected': True,
                   'confidence': 95,  # in %
                   'tolerance': 1,    # in %
                  }
    KPI = {'percentile': 50,
           'confidence': 75,
           'class': 'one-sided',
           'side': 'lower'}
    confidence_repeatability = 75


    # for now we write the case with list of csv files name

    ##
    # Preprocessing
    ##
    metric_all = []
    for series in data:
        metric_series = []
        for run in series:
            converged, metric_run = analysis_preprocessing(run,
                                                     metric,
                                                     plot=False,
                                                     convergence=convergence,
                                                     verbose=False)
            if converged is True:
                metric_series.append(metric_run)
            else:
                metric_series.append(np.nan)

        metric_all.append(metric_series)



    # artificial data for debugging
    metric_series = np.array([10,9,10,10,8,12,65,12,10])
    metric_all = [metric_series+0.1,
                  metric_series+0.3,
                  metric_series+0.07,
                  metric_series-0.2,
                  metric_series+0.13,
                  metric_series-0.34]

#     print(metric_all)

    ##
    # Performance evaluation
    ##
    KPI_series = []
    for series in metric_all:
        stationary, KPI_bounds = analysis_kpi(series, KPI, verbose=False)
        if stationary is True:
            series.sort()
            if KPI['side'] == 'lower':
                KPI_series.append(series[KPI_bounds[1]])
            else:
                KPI_series.append(series[KPI_bounds[0]])
        else:
            KPI_series.append(np.nan)

    ##
    # Repeatability
    ##
    to_plot = None
    stationary, repeatability_mean_value, repeatability_bounds, repeatability_score = analysis_repeatability(KPI_series, confidence_repeatability, verbose=False, to_plot=to_plot)


    ##
    # Produce the performance report
    ##

    # these info should be in the meta_data
    # get inspiration from Pantheon json file
    protocol_name = 'BBR'
    network_name = 'PantheonXYZ'
    metric_label = 'Throughput'
    KPI_label = '95% CI on 85th percentile'
    metric_desc = 'description to fetch from some dictionary sdfsdfsdf  sdfsdf sdfsdfs sdfsdfsdfsdfsdf  sdfsdf sdfsdfs sdfsdfsdfsdfsdf  sdfsdf sdfsdfs sdfsdfsdfsdfsdf  sdfsdf sdfsdfs sdfsdfsdfsdfsdf  sdfsdf sdfsdfs sdfsdfsdfsdfsdf  sdfsdf sdfsdfs sdfsdfsdfsdfsdf  sdfsdf sdfsdfs sdfsdfsdfsdfsdf  sdfsdf sdfsdfs sdfsdf'

    serie_ids = range(len(data))
    serie_labels = []
    for ids in serie_ids:
        serie_labels.append('Serie '+str(ids))

    if KPI['side'] == 'lower':
        direction = 'less or equal to'
    else:
        direction = 'greater or equal to'

    analysis_output = ''
    analysis_output += '# ---------------------------------------------------------------- \n'
    analysis_output += '# TriScale Performance Report \n'
    analysis_output += '# ---------------------------------------------------------------- \n'
    analysis_output += 'Protocol \t%s\n' % (protocol_name)
    analysis_output += 'Network \t%s\n' % (network_name)
    analysis_output += 'Metric \t\t%s\n' % (metric_label)
    analysis_output += 'defined as \t%s\n' % (metric_desc)
    analysis_output += 'KPI\t\t%s\n' % (KPI_label)
    analysis_output += '# ---------------------------------------------------------------- \n'
    analysis_output += '\n'
    analysis_output += 'For the different series, with a confidence level of %g%%,\n' % KPI['confidence']
    analysis_output += 'the %g-th percentile of the %s metric \n' % (
       KPI['percentile'],
       metric_label)

    analysis_output += 'in a run of %s is %s\n' % (
       protocol_name,
       direction)
    analysis_output += '\n'

    analysis_output += '%s :\t%f\t%s\n' % (
        serie_labels[0],
        KPI_series[0],
        metric['unit'])
    for serie_cnt in range(1,len(data)):
        analysis_output += '%s :\t%f\n' % (
            serie_labels[serie_cnt],
            KPI_series[serie_cnt])

    analysis_output += '\n'

#     if repeatable:
#         flag_repeatability = ''
#     else:
#         flag_repeatability = 'NOT '

#     analysis_output += 'Given a tolerance of %g%%, these results are %s%g%%-repeatable.\n' %             (tolerance_repeatability,
#              flag_repeatability,
#              confidence_repeatability)

    analysis_output += '# ---------------------------------------------------------------- \n'
    analysis_output += '\n'

    if stationary:
        analysis_output += 'With a confidence level of %g%%, \n' % KPI['confidence']
        analysis_output += ('the evaluation of %s %s on %s results in\n'
                            % (protocol_name,
                               metric_label,
                               network_name))
        analysis_output += ('Value :\t\t    %f \t%s \n'
                            % (repeatability_mean_value, metric['unit']) )
        analysis_output += ('Rep. Score :\t+/- %f \n'
                            % (repeatability_score) )

#         analysis_output += 'the repeatability score is\n'
#         analysis_output += ('Rep. Score :\t+/- %f \t%s \n'
#                             % (repeatability_score, metric['unit']) )
#         analysis_output += ('Rep. Score :\t+/- %f \t%s \n'
#                             % (repeatability_bounds[1], metric['unit']) )
    else:
        analysis_output += 'to be handled\n'
    analysis_output += '\n'
    analysis_output += '# ----------------------------------------------------------------'

#     if print_output:
    print(analysis_output)

    return


def analysis(data,
                 percentile,
                 confidence_percentile,
                 bound_side=None,
                 confidence_repeatability=None,
                 tolerance_repeatability=None,
                 data_label=None,
                 serie_labels=None,
                 print_output=True):
    '''
    Perform the analysis of (a series of) data, as suggested by TriScale.

    The analysis comprises of multiple steps:
    - The stationarity of each data series is assessed using autocorralation.
    - A confidence interval for the specified percentile and confidence level is computed, for each data series.
    - The repeatability of the data series is tested.
    - A confidence interval for the specified percentile and confidence level is computed, for entire data set.


    Parameters
    ----------
    data : list of list, or list of 1d-np.array
    percentile: float
        Must be strictly between 0 and 100.
    bound_side: string. [Optional]
        'lower' or 'upper'
        Whether we search an upper- or lower-bound of 'percentile'.
        If None, value is picked based on 'percentile', i.e.,
         - 'lower' for percentile < 50
         - 'upper' for percentile >= 50
    confidence_percentile: float
        Confidence level for estimating the given percentile.
        Must be strictly between 0 and 100.
        -> TODO: single/double sided comment for the median?
    confidence_repeatability: float or None. [Optional]
        Confidence level for the repeatability test.
        Must be strictly between 0 and 100.
        If None, repeatability test is not performed.
    tolerance_repeatability: float or None. [Optional]
        Must be strictly between 0 and 100.
        Tolerance for the repeatability test.
        If None, repeatability test is not performed.
    data_label: string, or None. [Optional]
        Label for the analysed data.
    series_label: list of strings, or None. [Optional]
        Label for the individual data series.


    Returns
    -------

    To be added

    Notes
    -----

    To be completed.

    The implementation of `theilslopes` follows [1]_. The intercept is
    not defined in [1]_, and here it is defined as ``median(y) -
    medslope*median(x)``, which is given in [3]_. Other definitions of
    the intercept exist in the literature. A confidence interval for
    the intercept is not given as this question is not addressed in
    [1]_.

    References
    ----------
    .. [1] P.K. Sen, "Estimates of the regression coefficient based on Kendall's tau",
           J. Am. Stat. Assoc., Vol. 63, pp. 1379-1389, 1968.
    .. [2] H. Theil, "A rank-invariant method of linear and polynomial
           regression analysis I, II and III",  Nederl. Akad. Wetensch., Proc.
           53:, pp. 386-392, pp. 521-525, pp. 1397-1412, 1950.
    .. [3] W.L. Conover, "Practical nonparametric statistics", 2nd ed.,
           John Wiley and Sons, New York, pp. 493.

    Examples
    --------

    Relevant?

    >>> from scipy import stats
    >>> import matplotlib.pyplot as plt

    >>> x = np.linspace(-5, 5, num=150)
    >>> y = x + np.random.normal(size=x.size)
    >>> y[11:15] += 10  # add outliers
    >>> y[-5:] -= 7

    Compute the slope, intercept and 90% confidence interval.  For comparison,
    also compute the least-squares fit with `linregress`:

    >>> res = stats.theilslopes(y, x, 0.90)
    >>> lsq_res = stats.linregress(x, y)

    Plot the results. The Theil-Sen regression line is shown in red, with the
    dashed red lines illustrating the confidence interval of the slope (note
    that the dashed red lines are not the confidence interval of the regression
    as the confidence interval of the intercept is not included). The green
    line shows the least-squares fit for comparison.

    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> ax.plot(x, y, 'b.')
    >>> ax.plot(x, res[1] + res[0] * x, 'r-')
    >>> ax.plot(x, res[1] + res[2] * x, 'r--')
    >>> ax.plot(x, res[1] + res[3] * x, 'r--')
    >>> ax.plot(x, lsq_res[1] + lsq_res[0] * x, 'g-')
    >>> plt.show()
    '''


    todo = ''
    todo += '# ---------------------------------------------------------------- \n'
    todo += '# TODO \n'
    todo += '# ---------------------------------------------------------------- \n'
    todo += '- handle the intermediary outputs\n'
    todo += '- finish the doctring\n'
    todo += '- polish the plot (in particular, the subplot with all data series)\n'
    todo += '- write output to file\n'
    todo += '- iterate through the list of inputs to see if we have all we need\n'
    todo += '- include autocorellation plots? \n'
    todo += '# ---------------------------------------------------------------- \n'
    print('%s' % todo)


    ##
    # Input checks
    ##

    # For now, we assume the inputs are correct...

    serie_ids = range(len(data))
    serie_labels = []
    for ids in serie_ids:
        serie_labels.append('Serie '+str(ids))

    ##
    # Stationarity test
    ##
    serie_cnt = 0
    for series in data:

        # test
        stationary = stationarity_test(series)
        # plot autocorrelation
#         plot_autocorr(data)
        # output
        if stationary:
            print(('%s : Data appears i.i.d. (95%% confidence)')
                  % serie_labels[serie_cnt])
        else:
            print(('%s : Data appears NOT I.D.D. !\nAnalysis continues but results are not trustworthy...')
                  % serie_labels[serie_cnt])
        #increment
        serie_cnt += 1

    ##
    # Compute and plot the confidence interval of each series
    ##
#     CI_bounds = ci( data, percentile, confidence_percentile, plot=False, bound_side=bound_side)
#     print(CI_bounds)

    ##
    # Repeatability test
    ##

    # TODO: Correct for multiple k!

#     if confidence_repeatability and tolerance_repeatability:
    # - take the CI bounds are new data series
    # - compute the two-sided CI on the median, with the given confidence_repeatability
    # - test the width of that CI agains the tolerance_repeatability value
#         repeatable = repeatability_test( CI_bounds, confidence_repeatability, tolerance_repeatability )

    ##
    # Compute CI for the entire data set and plot
    ##
#     data_all = [item for sublist in data for item in sublist]
#     print(data_all)
#     CI_bounds_final = ci( [data_all], percentile, confidence_percentile, plot=True, bound_side=bound_side )

    ##
    # Produce the outputs
    ##
    repeatable=True
    protocol_name = 'BBR'
    network_name = 'PantheonXYZ'
    metric_label = 'Throughput'
    metric_desc = 'description to fetch from some dictionary sdfsdfsdf  sdfsdf sdfsdfs sdfsdfsdfsdfsdf  sdfsdf sdfsdfs sdfsdfsdfsdfsdf  sdfsdf sdfsdfs sdfsdfsdfsdfsdf  sdfsdf sdfsdfs sdfsdfsdfsdfsdf  sdfsdf sdfsdfs sdfsdfsdfsdfsdf  sdfsdf sdfsdfs sdfsdfsdfsdfsdf  sdfsdf sdfsdfs sdfsdfsdfsdfsdf  sdfsdf sdfsdfs sdfsdf'
    if bound_side == 'lower':
        direction = 'less or equal to'
    else:
        direction = 'greater or equal to'

    analysis_output = ''
    analysis_output += '# ---------------------------------------------------------------- \n'
    analysis_output += '# Etalon report \n'
    analysis_output += '# ---------------------------------------------------------------- \n'
    analysis_output += 'Protocol \t%s\n' % (protocol_name)
    analysis_output += 'Network \t%s\n' % (network_name)
    analysis_output += 'Metric \t\t%s\n' % (metric_label)
    analysis_output += 'defined as \t%s\n' % (metric_desc)
    analysis_output += '# ---------------------------------------------------------------- \n\n'
    analysis_output += 'For the different series, with a confidence level of %g%%,\n' % confidence_percentile
    analysis_output += 'the %g-th percentile of the %s in a run of %s is %s\n\n' % (percentile,
       metric_label,
       protocol_name,
       direction
       )
    for serie_cnt in range(len(data)):
        analysis_output += '%s :\t%f\n' % (serie_labels[serie_cnt], 0)

    if confidence_repeatability and tolerance_repeatability:
        analysis_output += '\n'
        if repeatable:
            flag_repeatability = ''
        else:
            flag_repeatability = 'NOT '

        analysis_output += 'Given a tolerance of %g%%, these results are %s%g%%-repeatable.\n' %             (tolerance_repeatability,
             flag_repeatability,
             confidence_repeatability)

    analysis_output += '# ---------------------------------------------------------------- \n\n'


    analysis_output += ('Considering all collected data, with a confidence level of %g%%,\n'
                        % confidence_percentile)
    analysis_output += ('the %g-th percentile of the %s distribution for %s on network %s is %s %f\n\n'
                        % (percentile,
                           metric_label,
                           protocol_name,
                           network_name,
                           direction,
                           0))
    analysis_output += '# ----------------------------------------------------------------'

    if print_output:
        print(analysis_output)

    return CI_bounds_final[0]
