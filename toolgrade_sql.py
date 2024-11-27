savant_query = """
    SELECT
        pfx_x,
        pfx_z,
        plate_x,
        plate_z,
        release_speed,
        launch_speed,
        launch_angle,
        home_team,
        batter,
        delta_run_exp,
        description,
        balls,
        strikes,
        CASE WHEN description IN {takes} THEN 0 ELSE 1 END AS decision,
        CASE WHEN description IN {contact} THEN 1 ELSE 0 END AS contact,
        CASE WHEN description IS 'called_strike' THEN 1 ELSE 0 END AS cStrike,
        balls || '-' || strikes AS count,
        bb_type || launch_speed_angle AS bb_barrels        
    FROM
        statcast
    WHERE
        game_type IN {game_types}
        AND game_date BETWEEN '{start_date}' AND '{end_date}';
        """
