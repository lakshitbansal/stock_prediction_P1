import numpy as np

import matplotlib.pyplot as plt

from stock_prediction import create_model, load_data
from parameters import *


def plot_graph(test_df):
    """
    This function plots true close price along with predicted close price
    with blue and red colors respectively
    """
    plt.plot(test_df[f'true_adjclose_{LOOKUP_STEP}'], c='b')
    plt.plot(test_df[f'adjclose_{LOOKUP_STEP}'], c='r')
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend(["Actual Price", "Predicted Price"])
    plt.show()


    # def get_final_df(model, data):
    #     import pandas as pd
    #     import numpy as np

    # # helpers for profit
    # buy_profit  = lambda current, pred_future, true_future: true_future - current if pred_future > current else 0
    # sell_profit = lambda current, pred_future, true_future: current - true_future if pred_future < current else 0

    # X_test = data["X_test"]
    # y_test = data["y_test"]

    # # predictions
    # y_pred = model.predict(X_test)

    # # inverse-scale to prices if needed
    # if SCALE:
    #     y_test = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(np.expand_dims(y_test, axis=0)))
    #     y_pred = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(y_pred))

    # # start from clean copy and remove duplicate-named columns
    # test_df = data["test_df"].copy()
    # test_df = test_df.loc[:, ~test_df.columns.duplicated()].copy()

    # # helper: guarantee a 1-D numeric Series for a named column (handles DataFrame case)
    # def to_numeric_series(df, colname):
    #     col = df[colname]
    #     if isinstance(col, pd.DataFrame):
    #         col = col.iloc[:, 0]  # take first if duplicates existed
    #     return pd.to_numeric(pd.Series(col.values, index=df.index), errors="coerce")

    # # assign predicted & true prices as proper Series aligned to index
    # test_df[f"adjclose_{LOOKUP_STEP}"] = pd.to_numeric(pd.Series(np.ravel(y_pred), index=test_df.index), errors="coerce")
    # test_df[f"true_adjclose_{LOOKUP_STEP}"] = pd.to_numeric(pd.Series(np.ravel(y_test), index=test_df.index), errors="coerce")

    # # ensure base 'adjclose' is a 1-D numeric Series
    # test_df["adjclose"] = to_numeric_series(test_df, "adjclose")

    # # drop any rows that became NaN in critical columns
    # test_df = test_df.dropna(subset=["adjclose", f"adjclose_{LOOKUP_STEP}", f"true_adjclose_{LOOKUP_STEP}"])

    # # sort by date and compute profits
    # test_df.sort_index(inplace=True)
    # final_df = test_df

    # final_df["buy_profit"] = list(map(
    #     buy_profit,
    #     final_df["adjclose"],
    #     final_df[f"adjclose_{LOOKUP_STEP}"],
    #     final_df[f"true_adjclose_{LOOKUP_STEP}"],
    # ))
    # final_df["sell_profit"] = list(map(
    #     sell_profit,
    #     final_df["adjclose"],
    #     final_df[f"adjclose_{LOOKUP_STEP}"],
    #     final_df[f"true_adjclose_{LOOKUP_STEP}"],
    # ))

    # return final_df

# def get_final_df(model, data):
#     import pandas as pd
#     import numpy as np

#     # helpers for profit
#     buy_profit  = lambda current, pred_future, true_future: true_future - current if pred_future > current else 0
#     sell_profit = lambda current, pred_future, true_future: current - true_future if pred_future < current else 0

#     X_test = data["X_test"]
#     y_test = data["y_test"]

#     # predictions
#     y_pred = model.predict(X_test)

#     # inverse-scale to prices if needed
#     if SCALE:
#         y_test = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(np.expand_dims(y_test, axis=0)))
#         y_pred = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(y_pred))

#     # start from clean copy and remove duplicate-named columns
#     test_df = data["test_df"].copy()
#     test_df = test_df.loc[:, ~test_df.columns.duplicated()].copy()

#     # --- ensure base 'adjclose' exists as a numeric 1-D series ---
#     def _ensure_adjclose(df: pd.DataFrame) -> pd.Series:
#         if "adjclose" in df.columns:
#             col = df["adjclose"]
#         elif "Adj Close" in df.columns:
#             col = df["Adj Close"]
#         elif "close" in df.columns:
#             col = df["close"]
#         else:
#             # fallback: pull from full df using aligned index
#             base = data["df"].loc[df.index]
#             if "adjclose" in base.columns:
#                 col = base["adjclose"]
#             elif "Adj Close" in base.columns:
#                 col = base["Adj Close"]
#             elif "close" in base.columns:
#                 col = base["close"]
#             else:
#                 raise KeyError("Could not find an adjclose/Adj Close/close column to use.")
#         # if a DataFrame slipped through, flatten to 1-D series
#         if isinstance(col, pd.DataFrame):
#             col = col.iloc[:, 0]
#         return pd.to_numeric(pd.Series(col.values, index=df.index), errors="coerce")

#     test_df["adjclose"] = _ensure_adjclose(test_df)

#     # --- add predicted & true future prices as aligned numeric series ---
#     pred_col = f"adjclose_{LOOKUP_STEP}"
#     true_col = f"true_adjclose_{LOOKUP_STEP}"
#     test_df[pred_col] = pd.to_numeric(pd.Series(np.ravel(y_pred), index=test_df.index), errors="coerce")
#     test_df[true_col] = pd.to_numeric(pd.Series(np.ravel(y_test), index=test_df.index), errors="coerce")

#     # drop rows that can’t be used (any NaNs in critical columns)
#     need_cols = ["adjclose", pred_col, true_col]
#     have_cols = [c for c in need_cols if c in test_df.columns]
#     if len(have_cols) < 3:
#         # give a helpful error showing what columns exist
#         raise KeyError(f"Missing required columns. Have: {list(test_df.columns)} ; Need: {need_cols}")
#     test_df = test_df.dropna(subset=have_cols)

#     # sort & compute profits
#     test_df.sort_index(inplace=True)
#     final_df = test_df

#     final_df["buy_profit"] = list(map(
#         buy_profit,
#         final_df["adjclose"],
#         final_df[pred_col],
#         final_df[true_col],
#     ))
#     final_df["sell_profit"] = list(map(
#         sell_profit,
#         final_df["adjclose"],
#         final_df[pred_col],
#         final_df[true_col],
#     ))
#     return final_df

def get_final_df(model, data):
    import numpy as np
    import pandas as pd

    # Predict on test set
    X_test = data["X_test"]
    y_test = data["y_test"]
    y_pred = model.predict(X_test)

    # Inverse scale if needed
    if SCALE:
        y_test = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(np.expand_dims(y_test, axis=0)))
        y_pred = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(y_pred))

    # Base prices: pull from the original df at the test_df index
    base_df = data["df"]              # original copy saved in load_data
    test_idx = data["test_df"].index  # index of the test split
    base_df = base_df.loc[:, ~base_df.columns.duplicated()].copy()

    # choose a base price column robustly
    for cand in ("adjclose", "Adj Close", "close", "Close"):
        if cand in base_df.columns:
            base_series = base_df.loc[test_idx, cand]
            break
    else:
        raise KeyError("No base price column found (looked for 'adjclose', 'Adj Close', 'close', 'Close').")

    # Flatten & align lengths
    base_vals = np.asarray(base_series.values).ravel()
    y_pred = np.asarray(y_pred).ravel()
    y_test = np.asarray(y_test).ravel()
    m = min(len(base_vals), len(y_pred), len(y_test))
    base_vals, y_pred, y_test = base_vals[:m], y_pred[:m], y_test[:m]
    test_idx = test_idx[:m]

    # Build a clean DataFrame with guaranteed columns
    pred_col = f"adjclose_{LOOKUP_STEP}"
    true_col = f"true_adjclose_{LOOKUP_STEP}"
    final_df = pd.DataFrame(
        {
            "adjclose": pd.to_numeric(base_vals, errors="coerce"),
            pred_col:   pd.to_numeric(y_pred, errors="coerce"),
            true_col:   pd.to_numeric(y_test, errors="coerce"),
        },
        index=test_idx,
    ).dropna(subset=["adjclose", pred_col, true_col]).sort_index()

    # Profits
    buy_profit  = lambda current, pred_future, true_future: true_future - current if pred_future > current else 0
    sell_profit = lambda current, pred_future, true_future: current - true_future if pred_future < current else 0
    final_df["buy_profit"]  = list(map(buy_profit,  final_df["adjclose"], final_df[pred_col], final_df[true_col]))
    final_df["sell_profit"] = list(map(sell_profit, final_df["adjclose"], final_df[pred_col], final_df[true_col]))
    return final_df

# def get_final_df(model, data):
#     import numpy as np
#     import pandas as pd

#     # helpers
#     buy_profit  = lambda current, pred_future, true_future: true_future - current if pred_future > current else 0
#     sell_profit = lambda current, pred_future, true_future: current - true_future if pred_future < current else 0

#     X_test = data["X_test"]
#     y_test = data["y_test"]

#     # predictions
#     y_pred = model.predict(X_test)

#     # inverse-scale to prices if needed
#     if SCALE:
#         y_test = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(np.expand_dims(y_test, axis=0)))
#         y_pred = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(y_pred))

#     # start with clean test_df (unique cols)
#     test_df = data["test_df"].copy()
#     test_df = test_df.loc[:, ~test_df.columns.duplicated()].copy()

#     # ---------- ensure we have base price column as 'adjclose' ----------
#     candidates = ["adjclose", "Adj Close", "close", "Close"]
#     base_col = next((c for c in candidates if c in test_df.columns), None)
#     if base_col is None:
#         # fallback: pull from the full df with aligned index
#         base_all = data["df"].loc[:, ~data["df"].columns.duplicated()].copy()
#         base_all = base_all.reindex(test_df.index)
#         base_col = next((c for c in candidates if c in base_all.columns), None)
#         if base_col is None:
#             raise KeyError(f"Could not find any of {candidates} in test_df or data['df']. "
#                            f"Available test_df cols: {list(test_df.columns)}; "
#                            f"data['df'] cols: {list(base_all.columns)}")
#         base_series = base_all[base_col]
#     else:
#         base_series = test_df[base_col]

#     # if a DataFrame slipped through, squeeze to Series
#     if isinstance(base_series, pd.DataFrame):
#         base_series = base_series.iloc[:, 0]

#     test_df["adjclose"] = pd.to_numeric(pd.Series(base_series.values, index=test_df.index), errors="coerce")

#     # ---------- add prediction/true columns with aligned length ----------
#     n = len(test_df)
#     y_pred_flat = np.ravel(y_pred)
#     y_test_flat = np.ravel(y_test)

#     # align lengths (truncate to min length)
#     m = min(n, len(y_pred_flat), len(y_test_flat))
#     if m < n:
#         test_df = test_df.iloc[:m]
#     y_pred_flat = y_pred_flat[:m]
#     y_test_flat = y_test_flat[:m]

#     pred_col = f"adjclose_{LOOKUP_STEP}"
#     true_col = f"true_adjclose_{LOOKUP_STEP}"

#     test_df[pred_col] = pd.to_numeric(pd.Series(y_pred_flat, index=test_df.index), errors="coerce")
#     test_df[true_col] = pd.to_numeric(pd.Series(y_test_flat, index=test_df.index), errors="coerce")

#     # ---------- drop unusable rows & compute profits ----------
#     test_df = test_df.dropna(subset=["adjclose", pred_col, true_col]).sort_index()

#     final_df = test_df.copy()
#     final_df["buy_profit"] = list(map(buy_profit,  final_df["adjclose"], final_df[pred_col], final_df[true_col]))
#     final_df["sell_profit"] = list(map(sell_profit, final_df["adjclose"], final_df[pred_col], final_df[true_col]))

#     return final_df

# def get_final_df(model, data):
#     import pandas as pd
#     import numpy as np

#     # helpers for profit
#     buy_profit  = lambda current, pred_future, true_future: true_future - current if pred_future > current else 0
#     sell_profit = lambda current, pred_future, true_future: current - true_future if pred_future < current else 0

#     X_test = data["X_test"]
#     y_test = data["y_test"]

#     # predictions
#     y_pred = model.predict(X_test)

#     # inverse-scale to prices if needed
#     if SCALE:
#         y_test = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(np.expand_dims(y_test, axis=0)))
#         y_pred = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(y_pred))

#     # start from clean copy and remove duplicate-named columns
#     test_df = data["test_df"].copy()
#     test_df = test_df.loc[:, ~test_df.columns.duplicated()].copy()

#     # helper: guarantee a 1-D numeric Series for a named column (handles DataFrame case)
#     def to_numeric_series(df, colname):
#         col = df[colname]
#         if isinstance(col, pd.DataFrame):
#             col = col.iloc[:, 0]  # take first if duplicates existed
#         return pd.to_numeric(pd.Series(col.values, index=df.index), errors="coerce")

#     # assign predicted & true prices as proper Series aligned to index
#     test_df[f"adjclose_{LOOKUP_STEP}"] = pd.to_numeric(pd.Series(np.ravel(y_pred), index=test_df.index), errors="coerce")
#     test_df[f"true_adjclose_{LOOKUP_STEP}"] = pd.to_numeric(pd.Series(np.ravel(y_test), index=test_df.index), errors="coerce")

#     # ensure base 'adjclose' is a 1-D numeric Series
#     test_df["adjclose"] = to_numeric_series(test_df, "adjclose")

#     # drop any rows that became NaN in critical columns
#     test_df = test_df.dropna(subset=["adjclose", f"adjclose_{LOOKUP_STEP}", f"true_adjclose_{LOOKUP_STEP}"])

#     # sort by date and compute profits
#     test_df.sort_index(inplace=True)
#     final_df = test_df

#     final_df["buy_profit"] = list(map(
#         buy_profit,
#         final_df["adjclose"],
#         final_df[f"adjclose_{LOOKUP_STEP}"],
#         final_df[f"true_adjclose_{LOOKUP_STEP}"],
#     ))
#     final_df["sell_profit"] = list(map(
#         sell_profit,
#         final_df["adjclose"],
#         final_df[f"adjclose_{LOOKUP_STEP}"],
#         final_df[f"true_adjclose_{LOOKUP_STEP}"],
#     ))

#     return final_df

# def get_final_df(model, data):
#     """
#     This function takes the `model` and `data` dict to
#     construct a final dataframe that includes the features along
#     with true and predicted prices of the testing dataset
#     """
#     # if predicted future price is higher than the current,
#     # then calculate the true future price minus the current price, to get the buy profit
#     buy_profit  = lambda current, pred_future, true_future: true_future - current if pred_future > current else 0
#     # if the predicted future price is lower than the current price,
#     # then subtract the true future price from the current price
#     sell_profit = lambda current, pred_future, true_future: current - true_future if pred_future < current else 0
#     X_test = data["X_test"]
#     y_test = data["y_test"]
#     # perform prediction and get prices
#     y_pred = model.predict(X_test)
#     y_pred = model.predict(X_test)
#     if SCALE:
#         y_test = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(np.expand_dims(y_test, axis=0)))
#         y_pred = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(y_pred))

#     test_df = data["test_df"].copy()

#     # add predicted & true future prices
#     test_df[f"adjclose_{LOOKUP_STEP}"] = y_pred
#     test_df[f"true_adjclose_{LOOKUP_STEP}"] = y_test

#     # ensure numeric (avoid 'object'/str dtypes from any source)
#     import pandas as pd
#     test_df["adjclose"] = pd.to_numeric(test_df["adjclose"], errors="coerce")
#     test_df[f"adjclose_{LOOKUP_STEP}"] = pd.to_numeric(test_df[f"adjclose_{LOOKUP_STEP}"], errors="coerce")
#     test_df[f"true_adjclose_{LOOKUP_STEP}"] = pd.to_numeric(test_df[f"true_adjclose_{LOOKUP_STEP}"], errors="coerce")

#     # optional: handle any NaNs that slipped in
#     test_df = test_df.dropna(subset=["adjclose", f"adjclose_{LOOKUP_STEP}", f"true_adjclose_{LOOKUP_STEP}"])

#     # sort by date (as before)
#     test_df.sort_index(inplace=True)

#     # profit calcs (unchanged)
#     buy_profit  = lambda current, pred_future, true_future: true_future - current if pred_future > current else 0
#     sell_profit = lambda current, pred_future, true_future: current - true_future if pred_future < current else 0

#     final_df = test_df
#     final_df["buy_profit"] = list(map(
#         buy_profit,
#         final_df["adjclose"],
#         final_df[f"adjclose_{LOOKUP_STEP}"],
#         final_df[f"true_adjclose_{LOOKUP_STEP}"],
#     ))
#     final_df["sell_profit"] = list(map(
#         sell_profit,
#         final_df["adjclose"],
#         final_df[f"adjclose_{LOOKUP_STEP}"],
#         final_df[f"true_adjclose_{LOOKUP_STEP}"],
#     ))
#     return final_df
    # if SCALE:
    #     y_test = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(np.expand_dims(y_test, axis=0)))
    #     y_pred = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(y_pred))
    # test_df = data["test_df"]
    # # add predicted future prices to the dataframe
    # test_df[f"adjclose_{LOOKUP_STEP}"] = y_pred
    # # add true future prices to the dataframe
    # test_df[f"true_adjclose_{LOOKUP_STEP}"] = y_test
    # # sort the dataframe by date
    # test_df.sort_index(inplace=True)
    # final_df = test_df
    # # add the buy profit column
    # final_df["buy_profit"] = list(map(buy_profit,
    #                                 final_df["adjclose"],
    #                                 final_df[f"adjclose_{LOOKUP_STEP}"],
    #                                 final_df[f"true_adjclose_{LOOKUP_STEP}"])
    #                                 # since we don't have profit for last sequence, add 0's
    #                                 )
    # # add the sell profit column
    # final_df["sell_profit"] = list(map(sell_profit,
    #                                 final_df["adjclose"],
    #                                 final_df[f"adjclose_{LOOKUP_STEP}"],
    #                                 final_df[f"true_adjclose_{LOOKUP_STEP}"])
    #                                 # since we don't have profit for last sequence, add 0's
    #                                 )
    # return final_df


def predict(model, data):
    # retrieve the last sequence from data
    last_sequence = data["last_sequence"][-N_STEPS:]
    # expand dimension
    last_sequence = np.expand_dims(last_sequence, axis=0)
    # get the prediction (scaled from 0 to 1)
    prediction = model.predict(last_sequence)
    # get the price (by inverting the scaling)
    if SCALE:
        predicted_price = data["column_scaler"]["adjclose"].inverse_transform(prediction)[0][0]
    else:
        predicted_price = prediction[0][0]
    return predicted_price


# load the data
data = load_data(ticker, N_STEPS, scale=SCALE, split_by_date=SPLIT_BY_DATE,
                shuffle=SHUFFLE, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE,
                feature_columns=FEATURE_COLUMNS)

# construct the model
model = create_model(N_STEPS, len(FEATURE_COLUMNS), loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                    dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)

# load optimal model weights from results folder
model_path = os.path.join("results", model_name) + ".weights.h5"
model.load_weights(model_path)

# evaluate the model
loss, mae = model.evaluate(data["X_test"], data["y_test"], verbose=0)
# calculate the mean absolute error (inverse scaling)
if SCALE:
    mean_absolute_error = data["column_scaler"]["adjclose"].inverse_transform([[mae]])[0][0]
else:
    mean_absolute_error = mae

# get the final dataframe for the testing set
final_df = get_final_df(model, data)
# predict the future price
future_price = predict(model, data)
# we calculate the accuracy by counting the number of positive profits
accuracy_score = (len(final_df[final_df['sell_profit'] > 0]) + len(final_df[final_df['buy_profit'] > 0])) / len(final_df)
# calculating total buy & sell profit
total_buy_profit  = final_df["buy_profit"].sum()
total_sell_profit = final_df["sell_profit"].sum()
# total profit by adding sell & buy together
total_profit = total_buy_profit + total_sell_profit
# dividing total profit by number of testing samples (number of trades)
profit_per_trade = total_profit / len(final_df)
# printing metrics
print(f"Future price after {LOOKUP_STEP} days is {future_price:.2f}$")
print(f"{LOSS} loss:", loss)
print("Mean Absolute Error:", mean_absolute_error)
print("Accuracy score:", accuracy_score)
print("Total buy profit:", total_buy_profit)
print("Total sell profit:", total_sell_profit)
print("Total profit:", total_profit)
print("Profit per trade:", profit_per_trade)
# plot true/pred prices graph
plot_graph(final_df)
print(final_df.tail(10))
# save the final dataframe to csv-results folder
csv_results_folder = "csv-results"
if not os.path.isdir(csv_results_folder):
    os.mkdir(csv_results_folder)
csv_filename = os.path.join(csv_results_folder, model_name + ".csv")
final_df.to_csv(csv_filename)
