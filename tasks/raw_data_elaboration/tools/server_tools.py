import psycopg2
import pandas as pd
# https://www.postgresqltutorial.com/postgresql-tutorial/postgresql-order-by/
# https://www.postgresqltutorial.com/postgresql-tutorial/postgresql-where/
# https://www.psycopg.org/docs/usage.html


def _make_connection(USER, PASSWORD, HOST, PORT, DBNAME):
    cnx = psycopg2.connect(user = USER,
                           password = PASSWORD,
                           host = HOST,
                           port = PORT,
                           database = DBNAME)
    cursor_cnx = cnx.cursor()
    return cnx, cursor_cnx


def _close_connection(cnx, cursor_cnx):
    cursor_cnx.close()
    cnx.close()

def get_database_info(database, USER, PASSWORD, HOST, PORT, DBNAME):
    cnx = psycopg2.connect(user = USER,
                           password = PASSWORD,
                           host = HOST,
                           port = PORT,
                           database = DBNAME)

    cursor_cnx = cnx.cursor()

    query = "SELECT * FROM " + database
    #query = "SELECT * FROM INFORMATION_SCHEMA.COLUMNS --WHERE TABLE_NAME = data_dendro_lm LIMIT 10"

    with cursor_cnx as cursor:
        output = []
        cursor.execute(query)
        colnames = [desc[0] for desc in cursor.description]
        result = cursor.fetchall()
        for row in result:
            output.append(row)
        
    
    with open(database+'_column_names.txt', 'w+') as f:
        # write elements of list
        for item in colnames:
            f.write('%s\n' %item)

    f.close()

    _close_connection(cnx, cursor_cnx)
    


def get_metadata(USER, PASSWORD, HOST, PORT, DBNAME):
    cnx, cursor_cnx = _make_connection(USER, PASSWORD, HOST, PORT, DBNAME)
    query = ('SELECT measure_point, series_id,'
             'series_start,'
             'series_stop,'
             'series_cutout,'
             'series_active,'
             'series_display,'
             'variable_id,'
             'variable_name,'
             'variable_resolution,'
             'variable_units,'
             'sensor_id,'
             'sensor_name,'
             'sensor_class,'
             'sensor_data_source,'
             'position_id,'
             'series_height,'
             'series_exposition,'
             'series_distance_to_tree,'
             'tree_id,'
             'tree_name,'
             'tree_xcor,'
             'tree_ycor,'
             'tree_altitude,'
             'tree_genus,'
             'tree_species,'
             'tree_dbh,'
             'tree_height,'
             'tree_status,'
             'tree_age,'
             'tree_phloem_thickness_mm,'
             'tree_totalbark_thickness_mm,'
             'tree_sapwood_thickness_cm,'
             'tree_sapwood_area,'
             'series_dsri_max,'
             'series_dsri_min,'
             'series_twd_max_gp,'
             'series_twd_med_gp,'
             'series_twd_min_gp,'
             'series_twd_max_nogp,'
             'series_twd_med_nogp,'
             'series_twd_min_nogp,'
             'series_twd_max_frost,'
             'series_twd_med_frost,'
             'series_twd_min_frost,'
             'series_gro_start_doy_med,'
             'series_gro_end_doy_med,'
             'series_gro_max_yr,'
             'series_gro_med_yr,'
             'series_gro_min_yr,'
             'series_gro_max_month,'
             'series_gro_med_month,'
             'series_gro_min_month,'
             'series_gro_max_week,'
             'series_gro_med_week,'
             'series_gro_min_week,'
             'series_gro_max_day,'
             'series_gro_med_day,'
             'series_gro_min_day,'
             'series_gro_max_hr,'
             'series_gro_med_hr,'
             'series_gro_min_hr,'
             'series_timing_gro_year_max,'
             'series_timing_gro_year_min,'
             'series_timing_gro_month_max,'
             'series_timing_gro_week_max,'
             'series_timing_gro_doy_max,'
             'series_timing_gro_hour_max,'
             'series_timing_gro_hour_min,'
             'series_grohours_med,'
             'series_grohours_gp_med,'
             'series_grohours_percent_med,'
             'series_mds_gp_med,'
             'series_mds_gp_max,'
             'series_twd_gp_morning,'
             'series_proc_tol,'
             'series_proc_tol_out,'
             'series_proc_tol_jump,'
             'series_proc_frost_thr,'
             'site_id,'
             'site_name,'
             'site_subplot,'
             'site_exposition,'
             'site_xcor,'
             'site_ycor,'
             'site_altitude,'
             'site_forest_type,'
             'site_soil_type,'
             'site_affiliation,'
             'site_area,'
             'region,'
             'country,'
             'site_annual_temp,'
             'site_growth_temp,'
             'site_annual_precip,'
             'site_growth_precip,'
             'site_annual_rad,'
             'site_growth_rad,'
             'site_annual_relh,'
             'site_growth_relh,'
             'site_annual_vpd,'
             'site_growth_vpd,'
             'site_n_depo,'
             'site_ozon,'
             'site_nfk,'
             'site_temp_ref FROM view_metadata ORDER BY series_id')
    # Note: The query will produce a list of variables that appear in the order in which they are written.

    with cursor_cnx as cursor:
        cursor.execute(query)
        result = cursor.fetchall()

    keys = []
    for el in cursor.description:
        keys.append(el[0])

    _close_connection(cnx, cursor_cnx)
    return pd.DataFrame(data=result, columns=keys)


def get_data_element(value, query, USER, PASSWORD, HOST, PORT, DBNAME):
    cnx, cursor_cnx = _make_connection(USER, PASSWORD, HOST, PORT, DBNAME)

    with cursor_cnx as cursor:
        cursor.execute(query, value)
        result = cursor.fetchall()

    keys = []
    for el in cursor.description:
        keys.append(el[0])

    _close_connection(cnx, cursor_cnx)
    return pd.DataFrame(data=result, columns=keys)


if __name__ == "__main__":
    meta = get_metadata()
    print(meta)