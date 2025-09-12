# ---------- FORECASTING ----------
with tabs[3]:
    st.subheader("Monte Carlo Forecasting (Throughput Based)")

    if view_df.empty or view_df["_bucketer"].dropna().empty:
        st.info("No completion data available for forecasting.")
    else:
        # Build throughput history (items per week)
        throughput = (
            view_df["_bucketer"]
            .dropna()
            .dt.to_period("W")
            .value_counts()
            .values
        )
        if len(throughput) == 0:
            st.info("Not enough throughput data.")
        else:
            mode = st.radio(
                "Forecast mode",
                ["How many items in N weeks?", "When will X items be done?"],
                index=0,
            )
            start_date = st.date_input("Start date", datetime.date.today())

            if mode == "How many items in N weeks?":
                weeks = st.number_input("Weeks", min_value=1, max_value=52, value=4)
                sims = st.number_input(
                    "Simulations", min_value=1000, max_value=50000, value=10000, step=1000
                )
                results = []
                for _ in range(sims):
                    total_done = 0
                    for _ in range(weeks):
                        total_done += np.random.choice(throughput)
                    results.append(total_done)

                p50, p85, p95 = np.percentile(results, [50, 85, 95])

                st.write(f"In {weeks} weeks (starting {start_date:%d %b %Y}):")
                st.write(f"- **50% likely**: {int(p50)} items")
                st.write(f"- **85% likely**: {int(p85)} items")
                st.write(f"- **95% likely**: {int(p95)} items")

                # Histogram
                chart_data = pd.DataFrame({"Items Delivered": results})
                hist = (
                    alt.Chart(chart_data)
                    .mark_bar()
                    .encode(
                        alt.X("Items Delivered:Q", bin=alt.Bin(maxbins=30)),
                        y="count()",
                    )
                )
                st.altair_chart(hist, use_container_width=True)

            else:
                items = st.number_input(
                    "Number of items", min_value=1, max_value=200, value=10
                )
                sims = st.number_input(
                    "Simulations", min_value=1000, max_value=50000, value=10000, step=1000
                )
                results = []
                for _ in range(sims):
                    total_done = 0
                    week_count = 0
                    while total_done < items:
                        total_done += np.random.choice(throughput)
                        week_count += 1
                    results.append(week_count * 7)  # days

                p50, p85, p95 = np.percentile(results, [50, 85, 95])

                st.write(f"To deliver {items} items (starting {start_date:%d %b %Y}):")
                st.write(
                    f"- **50% likely**: {(start_date + datetime.timedelta(days=p50)):%d %b %Y}"
                )
                st.write(
                    f"- **85% likely**: {(start_date + datetime.timedelta(days=p85)):%d %b %Y}"
                )
                st.write(
                    f"- **95% likely**: {(start_date + datetime.timedelta(days=p95)):%d %b %Y}"
                )

                # Histogram
                chart_data = pd.DataFrame({"Days to Complete": results})
                hist = (
                    alt.Chart(chart_data)
                    .mark_bar()
                    .encode(
                        alt.X("Days to Complete:Q", bin=alt.Bin(maxbins=30)),
                        y="count()",
                    )
                )
                st.altair_chart(hist, use_container_width=True)
