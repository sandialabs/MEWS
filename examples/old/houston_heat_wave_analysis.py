# -*- coding: utf-8 -*-
"""
Created on 12/29/2021

@author: dlvilla

This scripting and functions run MEWS and EnergyPlus to show differences
in peak load and total energy use for energy plus commercial office proto-type
due to increasing frequency and severity of heat waves between 2020 and 2050
for a range of IPCC scenarios in a probabilistic context.

The study is for Albuquerque. This routine is called from run_albuquerque_study.bat
An equivalent batch file will have to be generated for unix and/or mac/os cases
as well.

The script is run via three input arguments

--start_year=2025         A valid year sometime in the future. This will determine
                          how much heat waves are shifted.
--scenario=SSP5-8.5       Must be a valid IPCC scenario name
--random_seed=519419765   Just to be sure that results are replicable

The class "Input" contains several options to alter the behavior of the script
that will need adjustment

    ep_path : str : must provide a valid path to an EnergyPlus executable. If you do
               not have this, then it can be downloaded at energyplus.net
    post_process_path : str : a relative path from ep_path that indicates
                              where energyplus results files will reside.
    run_parallel : bool : Indicate whether the run will be made in parallel
                          using multiprocessing.Pool or single runs. Keep True
                          unless multiprocessing is failing or you are debugging
                          this speeds up things by at least an order of magnitude
    only_post_process : bool : Set to true if the study has already run but
                               you are just working on refining graphical output
    plot_locations : str : path to a file where output from the run will be
                           placed.
    num_realizations : int : the number of separate runs for each IPCC scenario
                           and start year that is input. The original study
                           used 100.

It is not advised that other inputs be changed.

"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import subprocess
import shutil
import warnings
from copy import deepcopy
import pickle as pkl
from warnings import warn

from matplotlib import rc

font = {"size": 18, "family": "Times New Roman"}
rc("font", **font)


class Input:
    """
    All input to this scripting is contained in this class so that the entire
    study can be adapted by only making changes here.

    """

    # plot options
    font = {"size": 18, "family": "Times New Roman"}
    rc("font", **font)

    # provide a path to energy plus' latest executable.
    ep_path = r"/ascldap/users/dlvilla/EnergyPlus/EnergyPlus-9.6.0/build/Products/"
    exe_name = "energyplus-9.6.0"
    post_process_path = os.path.join(ep_path, "PostProcess", "ReadVarsESO.exe")
    # much faster to run parallel but sometimes there are platform or changes
    # that cause this to fail.
    run_parallel = True

    num_realizations = 100

    only_post_process = True
    print_progress = True
    # DO NOT CHANGE THIS!
    main_path = os.path.dirname(os.path.realpath("__file__"))

    building_area = 4982.19  # m2

    study_results_name = "houston_study_results_rerun02052022"

    pre_string = "HoustonStudy"

    plot_locations = r"./from_python/plots/houston_rerun02052022"

    # mews
    station = {
        "summaries": os.path.join(
            "example_data", "ClimateZone2A_Houston", "Houston.csv"
        ),
        "norms": os.path.join(
            "example_data", "ClimateZone2A_Houston", "Houston_norms.csv"
        ),
    }

    base_weather_file_name = os.path.join(
        "USA_TX_Houston-Bush.Intercontinental.AP.722430_TMY3.epw"
    )

    scenarios = ["SSP370", "SSP245", "SSP585"]
    years = [2025, 2050, 2075, 2100]
    confidence_intervals = ["5%", "50%", "95%"]
    weather_files = []
    for year in years:
        for scenario in scenarios:
            for ci in confidence_intervals:
                weather_files.append(
                    base_weather_file_name[:-4] + scenario + "_" + str(year) + "_" + ci
                )
    # random_seed = 54564863

    # Energy Plus
    # ****************************************************************
    # DEFINE THE PATHS FOR THE IDFs AND THE EPWS.
    # Of course, the MEWS code needs to be previously run to create all the EPWS.
    # The the files can be copied from there in the simulation folders.
    path_mews_files_HW = os.path.join(
        main_path,
        "example_data",
        "ClimateZone2A_Houston",
        "results",
        "mews_epw_results",
    )
    # give an absolute path here.
    path_idf_files = [
        os.path.join(
            main_path,
            "example_data",
            "ClimateZone2A_Houston",
            "IECC_OfficeMedium_STD2018_Tampa.idf",
        )
    ]
    # Parameters to be set for the plots
    zn = ["CLASSROOM_BOT"]
    vn = ["Temperature"]

    def __init__(self):
        pass


"""

PARTS OF THE FOLLOWING IS CODE BY CARLO BIANCO FROM NREL. It has not been approved
for release onto GitHUB by NREL so you need to get his consent to post it.
"""


def execute_command(input0):
    df = subprocess.Popen(input0, stdout=subprocess.PIPE)
    output, err = df.communicate()

    # mesg1 = output.decode('utf-8').split('\n')[1]
    # mesg2 = output.decode('utf-8').split('\n')[-2]
    if not err is None:
        errmsg = err.decode("utf-8")
    else:
        errmsg = None
    if not errmsg is None:
        return False
    else:
        return True


def double_column_index_find(df, name1, name2=None):
    if name2 is None:
        ind = df[df[0] == name1].index
    else:
        ind = df[(df[0] == name1) & (df[1] == name2)].index
    if len(ind) > 1:
        raise ValueError("The index key should be completely unique")
    else:
        return ind[0]


def read_eplus_subtable(
    df,
    report_name,
    sub_report_name,
    scenario,
    realization,
    year,
    ipcc_confidence_interval,
    multi_index_names,
    dtypes=None,
    to_dates=None,
    rows_wanted=None,
):

    # TODO - put checks in to assure that read is correct.
    rep_ind = double_column_index_find(df, "REPORT:", report_name)
    sub_ind = double_column_index_find(df, sub_report_name)

    # first non NaN indicates the beginning of data
    start_ind = df.iloc[sub_ind:, 1][~df.iloc[sub_ind:, 1].isna()].index[0]
    # next NaN indicates the end of the table.
    end_ind = df.iloc[start_ind:, 1][df.iloc[start_ind:, 1].isna()].index[0]

    # first NaN - 1 is the end of the columns
    begin_col = 1
    # no need to subtract one because end of range is exclusive
    end_col = df.iloc[start_ind, begin_col:][
        df.iloc[start_ind, begin_col:].isna()
    ].index[0]

    columns = df.iloc[start_ind - 1, begin_col:end_col].values.astype(str)
    columns[0] = "Variable Name"

    if dtypes is None:
        if not to_dates is None:
            raise ValueError("If dtypes is None, then to_dates must be None as well!")
        dtypes = [object for i in range(len(columns))]
        to_dates = [False for i in range(len(columns))]
    else:
        if to_dates is None:
            to_dates = [False for i in range(len(columns))]
        if len(dtypes) != len(columns):
            raise ValueError(
                "The dtypes input must have the same number of "
                + "entries as the number of columns in the table!"
            )

    table_dict = {}

    for column, dtype, colind, to_date in zip(
        columns, dtypes, range(begin_col, end_col), to_dates
    ):
        table_dict[column] = df.iloc[start_ind:end_ind, colind].astype(dtype)
        if to_date:
            table_dict[column] = pd.to_datetime(
                table_dict[column], format="%d-%b-%H:%M"
            )

    table = pd.DataFrame(table_dict)
    table.index = table["Variable Name"]

    table.drop("Variable Name", axis=1, inplace=True)

    if not rows_wanted is None:
        table = table.loc[rows_wanted, :]

    # translate into a multi-index that will make it easy to stack these results
    # with the unique index needed.
    tup_list = [
        (varname, scenario, realization, year, ipcc_confidence_interval)
        for varname in table.index
    ]
    table.index = pd.MultiIndex.from_tuples(tup_list, names=multi_index_names)

    return table


def post_process_single_run(epw, idf):
    # this is in the context of a directory
    # all of these terms will change if the requested information in the IDF
    # file changes. A better method for post processing is probably warranted!
    df_raw = pd.read_csv("eplustbl.csv", names=range(88), skip_blank_lines=False)

    # THIS IS LIKELY TO CHANGE FOR OTHER FILES!
    scenario = epw.split("TMY3")[1].split("_")[0]
    realization = epw.split(".")[-2].split("r")[-1]
    year = epw.split("_")[-3]
    ci = epw.split("_")[-2]

    multi_index_names = [
        "Variable Name",
        "Scenario",
        "Realization",
        "Year",
        "Confidence Interval",
    ]

    elec_table = read_eplus_subtable(
        df_raw,
        "Energy Meters",
        "Annual and Peak Values - Electricity",
        scenario,
        realization,
        year,
        ci,
        multi_index_names,
        dtypes=["string", float, float, "string", float, "string"],
        to_dates=[False, False, False, True, False, True],
        rows_wanted=[
            "Electricity:Facility",
            "Cooling:Electricity",
            "Heating:Electricity",
        ],
    )
    gas_table = read_eplus_subtable(
        df_raw,
        "Energy Meters",
        "Annual and Peak Values - Natural Gas",
        scenario,
        realization,
        year,
        ci,
        multi_index_names,
        dtypes=["string", float, float, "string", float, "string"],
        to_dates=[False, False, False, True, False, True],
        rows_wanted=[
            "NaturalGas:Facility",
            "NaturalGas:HVAC",
            "WaterSystems:NaturalGas",
        ],
    )

    comfort_table = read_eplus_subtable(
        df_raw,
        "Annual Building Utility Performance Summary",
        "Comfort and Setpoint Not Met Summary",
        scenario,
        realization,
        year,
        ci,
        multi_index_names,
        dtypes=["string", float],
        to_dates=[False, False],
        rows_wanted=None,
    )

    coolsz_table = read_eplus_subtable(
        df_raw,
        "Component Sizing Summary",
        "Coil:Cooling:DX:TwoSpeed",
        scenario,
        realization,
        year,
        ci,
        multi_index_names,
        dtypes=["string", float, float, float, float, float, float],
        to_dates=None,
        rows_wanted=None,
    )

    heatsz_table = read_eplus_subtable(
        df_raw,
        "Component Sizing Summary",
        "Coil:Heating:Fuel",
        scenario,
        realization,
        year,
        ci,
        multi_index_names,
        dtypes=["string", float],
        to_dates=None,
        rows_wanted=None,
    )

    # more perhaps in the future but for now this may be enough!
    return [elec_table, gas_table, comfort_table, coolsz_table, heatsz_table]


def run_and_post_process_ep(tup):
    # simplify the input structure
    (
        idf,
        main_path,
        name_main_folder,
        epw,
        path_mews_files_HW,
        ep_path,
        post_process_path,
        zone_names,
        variable_names,
        list_EPWs_HW,
        zn,
        vn,
        idx,
        only_post_process,
        print_progress,
        exe_name,
    ) = tup

    idf_name = os.path.basename(idf)
    list_df_single_model = []
    os.chdir(os.path.join(main_path, name_main_folder))
    if not os.path.exists(os.path.join(main_path, name_main_folder, idf_name)):
        os.makedirs(idf_name)

    os.chdir(os.path.join(main_path, name_main_folder, idf_name))
    if not os.path.exists(epw):
        os.makedirs(epw)
    os.chdir(os.path.join(main_path, name_main_folder, idf_name, epw))
    # Here, let's start copying the files in the folder.
    # This part can be commented out to speed up the process.
    # COPY IDF

    if not only_post_process and not os.path.exists("eplustbl.csv"):
        shutil.copyfile(idf, "in.idf")
        # COPY EPW
        shutil.copyfile(os.path.join(path_mews_files_HW, epw), "in.epw")

        # Now let's run EP and ReadVarsESO to generate the CSV results
        run_succeeded = execute_command(
            [os.path.join(ep_path, exe_name), "-w", "in.epw", "in.idf"]
        )
    else:
        run_succeeded = os.path.exists("eplustbl.csv")

    if run_succeeded:
        results = post_process_single_run(epw, idf)
    else:
        results = None

    if print_progress:
        print("Run {0:d} is complete".format(idx))
        print("")

    return [run_succeeded, results]


class EnergyPlusWrapper(Input):
    def __init__(self, Inp):

        self.scenarios = Inp.scenarios
        self.start_years = Inp.years
        self.wfile_names = Inp.weather_files
        self.exe_name = Inp.exe_name

        # unpack to local script context
        main_path = self.main_path
        path_mews_files_HW = self.path_mews_files_HW
        path_idf_files = self.path_idf_files
        zn = self.zn
        vn = self.vn
        run_parallel = self.run_parallel

        os.chdir(main_path)

        if run_parallel:
            try:
                import multiprocessing as mp

                pool = mp.Pool(mp.cpu_count() - 1)
            except:
                warnings.warn(
                    "Setting up a pool for multi-processing failed! reverting to non-parallel run!"
                )
                run_parallel = False

        zone_names = "|".join(zn)
        variable_names = "|".join(vn)

        list_EPWs_HW = self.wfile_names

        list_IDFs = path_idf_files

        list_IDFs = sorted(list_IDFs)

        list_EPWs_HW = sorted(list_EPWs_HW)

        name_main_folder = "ep_results"
        if not os.path.exists(os.path.join(main_path, name_main_folder)):
            os.makedirs(os.path.join(main_path, name_main_folder))

        # Create all the nested folders for all the considered combinations of IDFs and EPWs
        results = {}
        idx = 0
        for idf in list_IDFs:
            # Now let's run each IDF with different weather scenarios
            for epw in list_EPWs_HW:
                for realization in range(Inp.num_realizations):
                    epw_r = epw + "_r" + str(realization) + ".epw"

                    idx += 1
                    key_name = (idf, epw)

                    tup = (
                        idf,
                        main_path,
                        name_main_folder,
                        epw_r,
                        path_mews_files_HW,
                        self.ep_path,
                        self.post_process_path,
                        zone_names,
                        variable_names,
                        list_EPWs_HW,
                        zn,
                        vn,
                        idx,
                        self.only_post_process,
                        self.print_progress,
                        self.exe_name,
                    )

                    if os.path.exists(os.path.join(path_mews_files_HW, epw_r)):
                        if run_parallel:

                            results[key_name] = pool.apply_async(
                                run_and_post_process_ep, args=[tup]
                            )
                        else:
                            results[key_name] = run_and_post_process_ep(tup)
                    else:
                        warn(
                            "Weatherfile Path does not exist? This may be ok if "
                            + "MEWS did not capture all years for all scenarios: \n\n"
                            + os.path.join(path_mews_files_HW, epw_r)
                        )

        # gather any results passed through python.
        if run_parallel:
            results_get = {}
            for tup, poolObj in results.items():
                try:
                    results_get[tup] = poolObj.get()
                except AttributeError:
                    raise AttributeError(
                        "The multiprocessing module will not"
                        + " handle lambda functions or any"
                        + " other locally defined functions!"
                    )

            pool.close()
            self.results = results_get
        else:
            self.results = results


class FinalPostProcess(Input):
    def __init__(self, objEP):
        # show error bared plots of various variables with scenario in the
        # legend, year as the x axis and variable as the y axis where perhaps
        # several variables can be added to a 2-yaxis type plot.
        if not os.path.exists("houston_study_results_step2.pkl"):
            df_list = self._stack_tables(objEP)

            df_stats_list = self._table_realization_stats(df_list)
            pkl.dump([df_stats_list], open("houston_study_results_step2.pkl", "wb"))
        else:
            df_stats_list = pkl.load(open("houston_study_results_step2.pkl", "rb"))[0]
        # 1000 goes from GJ to kWh
        GJ_to_kWh = 277.778
        W_to_kW = 0.001

        normf = [
            [self.building_area / GJ_to_kWh, self.building_area, self.building_area],
            [self.building_area / GJ_to_kWh, self.building_area, self.building_area],
            [1],
        ]
        numsig = [3, 3, 1]
        sig_names_for_plot = [
            [
                "Electricity (kWh/yr/m$^2$)",
                "Minimum Electricity (W/m$2$)",
                "Peak Electricity (W/m$2$)",
            ],
            [
                "Natural Gas (kWh/yr/m$^2$)",
                "Minimum Natural Gas (W/m$2$)",
                "Peak Natural Gas (W/m$2$)",
            ],
            ["Hours Criterion Not Met (hr/yr)"],
        ]
        var_title = {
            "Cooling:Electricity": "HVAC Electricity",
            "Electricity:Facility": "Total Electricity",
            "Heating:Electricity": "Electric Heating",
            "Time Setpoint Not Met During Occupied Cooling": "Cooling Setpoint Not Met",
            "Time Not Comfortable Based on Simple ASHRAE 55-2004": "Discomfort Simple ASHRAE 55-2004",
        }

        for df_li, norf, nsig, signame in zip(
            df_stats_list[:3], normf, numsig, sig_names_for_plot
        ):
            self._plot_stats_df_list(df_li, norf, nsig, signame, var_title)

    def _table_realization_stats(self, df_list):

        df_copy = deepcopy(df_list)

        df_stats_list = []
        for df in df_copy:
            row_list = []
            ind_list = []
            # drop timestamp or string columns
            for name, col in df.iteritems():
                if not "float" in str(col.dtype):
                    df.drop(name, axis=1, inplace=True)
            for var_name, var_df in df.groupby(level=0):
                for scenario, scen_df in var_df.groupby(level=1):
                    for year, year_df in scen_df.groupby(level=3):
                        mean_names = ",".join(
                            ["Mean " + name for name in year_df.columns]
                        )
                        std_names = ",".join(
                            ["Standard Deviation " + name for name in year_df.columns]
                        )
                        min_names = ",".join(
                            ["Min " + name for name in year_df.columns]
                        )
                        max_names = ",".join(
                            ["Max " + name for name in year_df.columns]
                        )
                        q025_names = ",".join(
                            ["2.5% Quantile " + name for name in year_df.columns]
                        )
                        q975_names = ",".join(
                            ["97.5% Quantile " + name for name in year_df.columns]
                        )
                        new_names = (
                            mean_names
                            + ","
                            + std_names
                            + ","
                            + min_names
                            + ","
                            + max_names
                            + ","
                            + q025_names
                            + ","
                            + q975_names
                        ).split(",")
                        row = pd.concat(
                            [
                                year_df.mean(),
                                year_df.std(),
                                year_df.min(),
                                year_df.max(),
                                year_df.quantile(0.025),
                                year_df.quantile(0.975),
                            ]
                        )
                        row.index = new_names
                        row_list.append(row)
                        ind_list.append((var_name, scenario, year))

            df_stats_list.append(
                pd.DataFrame(row_list, index=pd.MultiIndex.from_tuples(ind_list))
            )

        return df_stats_list

    def _stack_tables(self, objEP):

        results = objEP.results

        df_list = None

        for key, lis in results.items():
            run_succeeded = lis[0]
            if run_succeeded:
                res_list = lis[1]

                if df_list is None:
                    df_list = res_list

                else:
                    tab_list = []
                    for tab, addtab in zip(df_list, res_list):
                        tab_list.append(pd.concat([tab, addtab]))

                    df_list = tab_list

        return df_list

    def _plot_stats_df_list(self, df, normf, num_sig, signame, vtitle):

        num_var = 0
        for var_name, var_df in df.groupby(level=0):
            num_var += 1

        not_legend_plotted = True
        only_plot_next = False
        # plot by variable
        var = -1
        for var_name, var_df in df.groupby(level=0):
            var += 1

            for idx in range(num_sig):
                line_width = 34
                fig, ax = plt.subplots(1, 1)  # this must be = num_sig
                for scenario, scen_df in var_df.groupby(level=1):
                    # num_sig = int(scen_df.shape[1]/6) # THIS IS DEPENDENT on the number of statistics computed!

                    years = np.array([tup[2] for tup in scen_df.index])
                    stat = []
                    for idy in range(6):
                        stat.append(scen_df.iloc[:, idx + idy * num_sig])
                        stat[-1].index = years

                    mean = stat[0].values
                    p025 = stat[-2].values
                    p975 = stat[-1].values
                    color = next(ax._get_lines.prop_cycler)["color"]
                    ax.errorbar(
                        stat[0].index,
                        mean / normf[idx],
                        yerr=[(mean - p025) / normf[idx], (p975 - mean) / normf[idx]],
                        label=scenario,
                        elinewidth=line_width,
                        color=color,
                        linewidth=3,
                    )
                    ax.scatter(
                        stat[0].index,
                        stat[2].values / normf[idx],
                        marker="x",
                        color=color,
                    )
                    ax.scatter(
                        stat[0].index,
                        stat[3].values / normf[idx],
                        marker="x",
                        color=color,
                    )
                    ax.set_ylabel(signame[idx])

                    ax.grid("on")
                    line_width -= 8
                    if var_name in vtitle:
                        vname = vtitle[var_name]
                    else:
                        vname = var_name
                # ax.set_title(vname)
                if not_legend_plotted:
                    ax.legend(fontsize=12)
                    if only_plot_next:
                        not_legend_plotted = False
                fname = self.pre_string + "_" + vname + "_" + signame[idx] + ".png"
                for char in r"$\/^()[] :":
                    fname = fname.replace(char, "_")
                plt.tight_layout()
                plt.savefig(os.path.join(self.plot_locations, fname), dpi=1000)
            if not_legend_plotted:
                only_plot_next = True


if __name__ == "__main__":
    run_local = True

    ## Eventually get rid of this format. No need for *.bat
    # if run_local:
    #     scenarios = ["SSP5-8.5"]
    #     start_years = [2025]
    #     confidence_intervals = ["95%"]
    #     random_seed = 519419765
    # else:
    #     try:
    #        opts, args = getopt.getopt(sys.argv[1:],"s:y:r:c",["scenario=","start_year=","random_seed=","confidence_interval="])
    #     except getopt.GetoptError:
    #        print("")
    #        print('Input invalid')
    #        print("")
    #        sys.exit(2)

    #     if not opts:
    #        print("")
    #        print('No command line options given, you must have "--scenario=<IPCC scenario name>" "--start_year=<year>" "--random_seed=<number>" and "--confidence_interval=<ci>"')
    #        print("")
    #        sys.exit(2)

    #     for opt, arg in opts:
    #        if opt in ("-s","--scenario"):
    #             scenarios = [str(arg)]
    #        elif opt in ("-y","--start_year"):
    #             start_years = [int(arg)]
    #        elif opt in ("-r","--random_seed"):
    #            random_seed = int(arg)
    #        elif opt in ("-c","--confidence_interval"):
    #            confidence_intervals = [str(arg)]

    Inp = Input()

    if not Inp.only_post_process:  # not os.path.exists("study_results.pkl"):

        objEP = EnergyPlusWrapper(Inp)

        pkl.dump([objEP], open(Inp.study_results_name + ".pkl", "wb"))
    else:
        pass
        if not "objEP" in vars():
            [objEP] = pkl.load(open(Inp.study_results_name + ".pkl", "rb"))

    if Inp.only_post_process:
        plt.close("all")
        FinalPostProcess(objEP)
