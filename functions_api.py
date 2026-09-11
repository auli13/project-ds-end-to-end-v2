import pickle
import pandas as pd

with open('trained_models/trained_models.pkl', 'rb') as f:
    result = pickle.load(f)


def _closest_sensor(row, sensors: dict):
    # Find the sensor whose current reading is closest to its known failure threshold_mean.
    distances = {}
    for sensor_name, threshold_mean in sensors.items():
        curr_col = f'{sensor_name}_curr'
        if curr_col in row.index and threshold_mean is not None:
            distances[sensor_name] = abs(float(row[curr_col]) - threshold_mean)
    if not distances:
        return None, None
    prob_fail_sensor = min(distances, key=distances.get)
    return prob_fail_sensor, distances


def predict_station_risk(station: str) -> dict:
    # Get the probability of a Mechanical/Electrical failure in the next 30 minutes for a specific station.
    r = result[station]
    proba = float(r['model'].predict_proba(r['last_row'])[0, 1])
    return {'station': station, 'risk_next_30min': round(proba, 3), 'model_used': r['model_key']}


def get_prob_fail_sensor(station: str) -> dict:
    # Estimate which sensor is closest to its known failure threshold.
    r = result[station]
    row = r['last_row'].iloc[0]
    prob_fail_sensor, distances = _closest_sensor(row, r['sensors'])
    if prob_fail_sensor is None:
        return {'station': station, 'prob_fail_sensor': None}
    return {
        'station': station,
        'prob_fail_sensor': prob_fail_sensor,
        'distance_to_threshold': round(distances[prob_fail_sensor], 3),
        'all_distances': {k: round(v, 3) for k, v in sorted(distances.items(), key=lambda x: x[1])},
    }


def get_station_sensors(station: str) -> dict:
    # Get the list of sensors for ST.
    sensors = result[station]['sensors']
    return {'station': station, 'sensors': list(sensors.keys())}


def get_sensor_threshold(station: str, sensor: str) -> dict:
    # Get the detected failure threshold for a specific sensor at a specific ST."""
    sensors = result[station]['sensors']
    return {'station': station, 'sensor': sensor, 'threshold_mean': sensors.get(sensor)}


downtime_events = pd.read_csv('source/output/downtime_events.csv')


def get_downtime_history(station: str, cause: str = "") -> dict:
    # Get downtime history.
    df = downtime_events[downtime_events['station_id'] == station]
    if cause:
        df = df[df['cause'] == cause]
    return {
        'station': station, 'cause': cause or 'all', 'count': len(df),
        'mean_duration_min': round(df['minutes'].mean(), 1) if len(df) else None,
    }


def get_model_performance(station: str) -> dict:
    # Get the evaluation metrics of the model used for a station.
    r = result[station]
    metrics = r['eval']
    return {
        'station': station, 'model_used': r['model_key'],
        'recall': metrics['rec'], 'precision': metrics['prec'],
        'f1_score': metrics['f1'], 'overfitting_gap': metrics['gap'],
    }


def get_risk_at_step(station: str, step: int) -> dict:
    # Get the predicted risk and the real status at a specific minute.
    r = result[station]
    test_df = r['test_df']
    step = max(0, min(step, len(test_df) - 1))

    model = r['model']
    feature_cols = r['feature_cols']

    row = test_df.iloc[[step]]
    proba = float(model.predict_proba(row[feature_cols])[0, 1])

    threshold = r['threshold']
    predicted_failure = bool(proba >= threshold)

    # Predict which sensor could generate a failure
    prob_fail_sensor = None
    if predicted_failure:
        prob_fail_sensor, _ = _closest_sensor(row.iloc[0], r['sensors'])

    is_running = bool(row['is_running'].iloc[0])

    # trigger_sensor are only marked on the FIRST minute of a stop,
    # so walk backward through the current downtime block to find its true origin.
    stop_cause = None
    trigger_sensor = None
    if not is_running:
        i = step
        while i >= 0 and test_df['is_running'].iloc[i] == 0:
            for c in ['Mechanical', 'Electrical', 'Blocked', 'Starved', 'Changeover', 'Quality']:
                col = f'cause_{c}'
                if col in test_df.columns and test_df[col].iloc[i] == 1:
                    stop_cause = c
                    ts = test_df['trigger_sensor'].iloc[i] if 'trigger_sensor' in test_df.columns else None
                    trigger_sensor = ts if pd.notna(ts) else None  # NaN is truthy in Python, must check explicitly
                    break
            if stop_cause is not None:
                break
            i -= 1

    # target = 1 if a Mechanical/Electrical failure occurs in the next 30 real minutes (what the model predicts)
    actual_failure_next_30min = bool(row['target'].iloc[0]) if 'target' in row.columns else None

    return {
        'station': station, 'step': step, 'max_step': len(test_df) - 1,
        'timestamp': str(row['timestamp'].iloc[0]),
        'risk_next_30min': round(proba, 3), 'model_used': r['model_key'],
        'threshold': round(threshold, 3), 'predicted_failure': predicted_failure,
        'prob_fail_sensor': prob_fail_sensor,
        'is_running': is_running, 'stop_cause': stop_cause, 'trigger_sensor': trigger_sensor,
        'actual_failure_next_30min': actual_failure_next_30min,
    }


def get_step_from_datetime(dt) -> int:
    # Move the timeline slider for a given datetime.
    timestamps = result['ST1']['test_df']['timestamp']  # shared across all 8 stations
    dt = pd.Timestamp(dt)

    idx = timestamps.searchsorted(dt)
    idx = min(idx, len(timestamps) - 1)

    # snap to the closer neighbor (searchsorted gives the first value >= dt, the previous one may be closer)
    if idx > 0 and abs(timestamps.iloc[idx - 1] - dt) <= abs(timestamps.iloc[idx] - dt):
        idx = idx - 1

    return int(idx)


ALL_TOOLS_FUNCTIONS = {
    "predict_station_risk": predict_station_risk,
    "get_prob_fail_sensor": get_prob_fail_sensor,
    "get_station_sensors": get_station_sensors,
    "get_sensor_threshold": get_sensor_threshold,
    "get_downtime_history": get_downtime_history,
    "get_model_performance": get_model_performance,
}