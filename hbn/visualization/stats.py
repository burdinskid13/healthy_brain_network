def compare_slopes(df1, df2, alpha=0.05, x='age_round', y='percent'):
    """
    Compares the slopes of two lines and determines if the difference is statistically significant.

    Args:
        df1, df2: Pandas DataFrames with 'x' and 'y' columns.
        alpha: Significance level for the t-test.
        x, y: Column names for the x and y variables.

    Returns:
        A tuple containing:
            - The steeper line (either 'df1' or 'df2')
            - A boolean indicating whether the difference is statistically significant
    """
    import pandas as pd
    from scipy.stats import linregress, ttest_ind
    import statsmodels.formula.api as smf

    # Calculate slopes using linear regression
    slope1 = linregress(df1[x], df1[y]).slope
    slope2 = linregress(df2[x], df2[y]).slope

    print(f'Slopes: {slope1:.3f} vs. {slope2:.3f}')

def f_test(dataframe, dependent_col, factor1_col, factor2_col=None):
    """
    Performs ANOVA (one-way or two-way) and post-hoc Tukey HSD tests.

    Args:
        dataframe (pd.DataFrame): The input DataFrame.
        dependent_col (str): The name of the dependent variable column (numerical).
        factor1_col (str): The name of the first independent variable column (categorical).
        factor2_col (str, optional): The name of the second independent variable column (categorical).
                                     If None, a one-way ANOVA is performed. Defaults to None.

    Returns:
        None: Prints the ANOVA table and post-hoc results to the console.
    """
    import pandas as pd
    import statsmodels.formula.api as smf
    import statsmodels.stats.anova as anova
    import numpy as np
    from statsmodels.stats.multicomp import pairwise_tukeyhsd # Import for Tukey HSD
        
    if factor2_col:
        print(f"\n--- Performing Two-Way ANOVA for {dependent_col} by {factor1_col} and {factor2_col} ---")
        # Construct the ANOVA formula dynamically for two-way ANOVA
        formula = f"{dependent_col} ~ C({factor1_col}) * C({factor2_col})"
        analysis_type = "two-way"
    else:
        print(f"\n--- Performing One-Way ANOVA for {dependent_col} by {factor1_col} ---")
        # Construct the ANOVA formula dynamically for one-way ANOVA
        formula = f"{dependent_col} ~ C({factor1_col})"
        analysis_type = "one-way"

    model = smf.ols(formula, data=dataframe).fit()

    # Generate the ANOVA table
    anova_table = anova.anova_lm(model, typ=2) # typ=2 for Type II sum of squares, common for unbalanced designs
    print(anova_table)

    # --- ANOVA Interpretation ---
    print("\n--- ANOVA Interpretation ---")
    if analysis_type == "two-way":
        interaction_term = f'C({factor1_col}):C({factor2_col})'
        if interaction_term in anova_table.index and anova_table.loc[interaction_term, 'PR(>F)'] < 0.05:
            print(f"The interaction effect between {factor1_col} and {factor2_col} is statistically significant (p < 0.05).")
            print(f"This means the effect of {factor1_col} on {dependent_col} depends on the level of {factor2_col}, and vice-versa.")
            print("Therefore, post-hoc tests will focus on the combined groups.")
        else:
            print(f"The interaction effect between {factor1_col} and {factor2_col} is NOT statistically significant (p >= 0.05).")
            print(f"You may proceed to interpret the main effects of {factor1_col} and {factor2_col}.")
            print("Post-hoc tests will still be performed on combined groups for thoroughness,")
            print("but if only main effects were significant, you might consider post-hoc on main effects.")
    else: # One-way ANOVA
        if anova_table.loc[f'C({factor1_col})', 'PR(>F)'] < 0.05:
            print(f"The main effect of {factor1_col} is statistically significant (p < 0.05).")
            print(f"This indicates there is a significant difference in {dependent_col} across the levels of {factor1_col}.")
            print("Post-hoc tests will identify which specific group comparisons are significant.")
        else:
            print(f"The main effect of {factor1_col} is NOT statistically significant (p >= 0.05).")
            print(f"There is no significant difference in {dependent_col} across the levels of {factor1_col}.")
            print("Post-hoc tests are typically not performed if the overall ANOVA is not significant, but will be shown for completeness.")


    # --- Prepare data for Post-Hoc Tests ---
    # Create a new column that represents the combined groups for post-hoc
    df_for_posthoc = dataframe.copy()
    if factor2_col:
        df_for_posthoc['combined_group'] = df_for_posthoc[factor1_col].astype(str) + '-' + df_for_posthoc[factor2_col].astype(str)
    else:
        df_for_posthoc['combined_group'] = df_for_posthoc[factor1_col].astype(str)


    # --- Perform Post-Hoc Tests (Tukey HSD) using statsmodels ---
    print("\n--- Performing Post-Hoc Tests (Tukey HSD) using statsmodels ---")
    # The 'endog' parameter is the dependent variable (scores)
    # The 'groups' parameter is the categorical variable defining the groups
    posthoc_results = pairwise_tukeyhsd(endog=df_for_posthoc[dependent_col],
                                        groups=df_for_posthoc['combined_group'],
                                        alpha=0.05) # alpha is the significance level

    # Print the post-hoc results summary
    print(posthoc_results)

    # --- Interpret Post-Hoc Results ---
    print("\n--- Post-Hoc Interpretation ---")
    alpha = 0.05 # Significance level

    # The 'results_table' attribute of the TukeyHSDResults object contains the comparisons
    results_df = pd.DataFrame(data=posthoc_results._results_table.data[1:],
                              columns=posthoc_results._results_table.data[0])

    significant_comparisons = []
    for index, row in results_df.iterrows():
        group1 = row['group1']
        group2 = row['group2']
        p_value = row['p-adj']

        if p_value < alpha:
            significant_comparisons.append(f"Significant difference between {group1} and {group2} (p = {p_value:.3f})")

    if significant_comparisons:
        print("\nStatistically Significant Comparisons (p < 0.05):")
        for comp in significant_comparisons:
            print(comp)
    else:
        print("\nNo statistically significant pairwise differences found after post-hoc testing (at alpha=0.05).")


def perform_flexible_proportion_comparison(df, compare_col, count_col, overall_count_col, group_col=None, alpha=0.05, correction='holm-bonferroni'):
    """
    Performs two-proportion z-tests for all pairwise comparisons between
    the values of a specified comparison column, optionally within groups
    of a specified grouping column. Applies Standard Bonferroni and
    Holm-Bonferroni multiple testing corrections.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        compare_col (str): Name of the column whose values will be compared
                             (e.g., 'DX' or 'sex').
        count_col (str): Name of the column containing the count of events.
        overall_count_col (str): Name of the column containing the total count.
        group_col (str, optional): Name of the column to group the data by
                                     (e.g., 'sex' or 'DX'). Defaults to None,
                                     in which case comparisons are done across
                                     the entire DataFrame.
        alpha (float, optional): Significance level. Defaults to 0.05.
        correction (str, optional): Multiple testing correction method
                                     ('bonferroni' or 'holm-bonferroni').
                                     Defaults to 'holm-bonferroni'.

    Returns:
        pd.DataFrame: DataFrame with added columns for the compared groups,
                      z-statistic, p-value, Bonferroni corrected p-value and
                      rejection status, and Holm-Bonferroni corrected p-value
                      and rejection status.
    """
    import pandas as pd
    import statsmodels.api as sm
    import numpy as np
    from statsmodels.sandbox.stats.multicomp import multipletests
    from itertools import combinations

    results_list = []

    if group_col:
        unique_groups = df[group_col].unique()
        for group_value in unique_groups:
            df_group = df[df[group_col] == group_value]
            unique_comparisons = df_group[compare_col].unique()

            for comp1, comp2 in combinations(unique_comparisons, 2):
                try:
                    data1 = df_group[df_group[compare_col] == comp1].iloc[0]
                    data2 = df_group[df_group[compare_col] == comp2].iloc[0]

                    sample1 = data1[count_col]
                    total1 = data1[overall_count_col]
                    sample2 = data2[count_col]
                    total2 = data2[overall_count_col]

                    count = np.array([sample1, sample2])
                    nobs = np.array([total1, total2])
                    z_stat, p_value = sm.stats.proportions_ztest(count, nobs, alternative='two-sided')

                    results_list.append({
                        group_col: group_value,
                        f'{compare_col}_1': comp1,
                        f'{compare_col}_2': comp2,
                        'z_statistic': z_stat,
                        'p_value': p_value,
                        f'{compare_col}_1_count': sample1,
                        f'{compare_col}_1_total': total1,
                        f'{compare_col}_2_count': sample2,
                        f'{compare_col}_2_total': total2
                    })
                except IndexError:
                    print(f"Warning: Insufficient data for comparison between '{comp1}' and '{comp2}' within group '{group_value}'. Skipping.")
                except Exception as e:
                    print(f"An error occurred during comparison between '{comp1}' and '{comp2}' within group '{group_value}': {e}")
        grouping_str = f" (Grouped by '{group_col}')"
    else:
        unique_comparisons = df[compare_col].unique()
        for comp1, comp2 in combinations(unique_comparisons, 2):
            try:
                data1 = df[df[compare_col] == comp1].iloc[0]
                data2 = df[df[compare_col] == comp2].iloc[0]

                sample1 = data1[count_col]
                total1 = data1[overall_count_col]
                sample2 = data2[count_col]
                total2 = data2[overall_count_col]

                count = np.array([sample1, sample2])
                nobs = np.array([total1, total2])
                z_stat, p_value = sm.stats.proportions_ztest(count, nobs, alternative='two-sided')

                results_list.append({
                    f'{compare_col}_1': comp1,
                    f'{compare_col}_2': comp2,
                    'z_statistic': z_stat,
                    'p_value': p_value,
                    f'{compare_col}_1_count': sample1,
                    f'{compare_col}_1_total': total1,
                    f'{compare_col}_2_count': sample2,
                    f'{compare_col}_2_total': total2
                })
            except IndexError:
                print(f"Warning: Insufficient data for comparison between '{comp1}' and '{comp2}'. Skipping.")
            except Exception as e:
                print(f"An error occurred during comparison between '{comp1}' and '{comp2}': {e}")
        grouping_str = ""

    results_df = pd.DataFrame(results_list)
    if results_df.empty:
        print("No valid comparisons could be made based on the provided data.")
        return results_df

    p_values = results_df['p_value'].values
    num_tests = len(p_values)

    # Bonferroni Correction
    reject_bonferroni, p_corrected_bonferroni, _, _ = multipletests(p_values, alpha=alpha, method='bonferroni')
    results_df['p_corrected_bonferroni'] = p_corrected_bonferroni
    results_df['reject_null_bonferroni'] = reject_bonferroni

    if correction == 'bonferroni':
        print(f"\n--- Results with Standard Bonferroni Correction{grouping_str} ---")
        for index, row in results_df.iterrows():
            comparison_label = f"{row[f'{compare_col}_1']} vs. {row[f'{compare_col}_2']}"
            if group_col:
                comparison_label += f" (within {group_col} = '{row[group_col]}')"
            print(f"\n--- Comparison: {comparison_label} ---")
            print(f"Z-statistic: {row['z_statistic']:.3f}")
            print(f"Uncorrected P-value: {row['p_value']:.3f}")
            print(f"Bonferroni Corrected P-value: {row['p_corrected_bonferroni']:.3f}")
            if row['reject_null_bonferroni']:
                print(f"Reject null hypothesis at alpha={alpha}.")
                print(f"There is a significant difference in rates between {row[f'{compare_col}_1']} and {row[f'{compare_col}_2']}.")
            else:
                print(f"Fail to reject null hypothesis at alpha={alpha}.")
                print(f"There is no significant difference in rates between {row[f'{compare_col}_1']} and {row[f'{compare_col}_2']}.")

    if correction == 'holm-bonferroni':
        # Holm-Bonferroni Correction
        reject_holm, p_corrected_holm, _, _ = multipletests(p_values, alpha=alpha, method='holm')
        results_df['p_corrected_holm'] = p_corrected_holm
        results_df['reject_null_holm'] = reject_holm

        print(f"\n--- Results with Holm-Bonferroni Correction{grouping_str} ---")
        for index, row in results_df.iterrows():
            comparison_label = f"{row[f'{compare_col}_1']} vs. {row[f'{compare_col}_2']}"
            if group_col:
                comparison_label += f" (within {group_col} = '{row[group_col]}')"
            print(f"\n--- Comparison: {comparison_label} ---")
            print(f"Z-statistic: {row['z_statistic']:.3f}")
            print(f"Uncorrected P-value: {row['p_value']:.3f}")
            print(f"Holm-Bonferroni Corrected P-value: {row['p_corrected_holm']:.3f}")
            if row['reject_null_holm']:
                print(f"Reject null hypothesis at alpha={alpha}.")
                print(f"There is a significant difference in rates between {row[f'{compare_col}_1']} and {row[f'{compare_col}_2']}.")
            else:
                print(f"Fail to reject null hypothesis at alpha={alpha}.")
                print(f"There is no significant difference in rates between {row[f'{compare_col}_1']} and {row[f'{compare_col}_2']}.")

    return results_df
