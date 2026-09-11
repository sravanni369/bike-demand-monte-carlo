My Monte Carlo plan cut the penalty by 40%. Then I found that one line of NumPy did the same thing.

I took section 3.9.5 of Deep Learning by Ian Goodfellow, Yoshua Bengio and Aaron Courville (MIT Press, printed pages 65 to 66): the empirical distribution. Every observation gets probability mass 1/m. Those pages have equations, not code. The 50 lines of Python are my implementation of Eq. 3.28 and the Monte Carlo expectation from section 17.1.2.

Then I gave it a real shop. UCI Bike Sharing: 731 days of Capital Bikeshare rentals, 2011 to 2012. For each day of 2012 the code sees only the previous 90 days and plans tomorrow's rental capacity. I assumed falling short costs four times as much as idle capacity, and said so.

Against a rolling-mean plan, the Monte Carlo plan cut the mean penalty by 40.56% over 366 days. 5,000 draws per decision, seeded, two runs byte-identical.

The audit is where it got useful.

For a 4:1 loss the best plan is the 80th percentile of the window. np.quantile(history, 0.8) scored 1,946. Monte Carlo scored 1,947. The draws added nothing. The loss-aware target did all the work.

Then the quarters. Q1 +41%, Q2 +63%, Q3 +55%, Q4 minus 34%. When demand fell in the autumn, a window full of summer days over-planned and lost to the plain mean. The annual number hid a losing quarter.

And the window mattered more than the method. The same quantile rule over 7 days scored 59% better than the mean. I ran that after the fact, so it goes in the README as a sensitivity, not a headline.

Covered 64% of days against an 80% target. The gap is the series rising, not a bug.

What changed for me: I now run the one-line version before I trust the simulation, and I break every annual number into quarters before it goes anywhere near a post.

Code, the 50-line script, every log, the VS Code screenshot and the audit baselines are on my GitHub if you want to check the numbers: github.com/sravanni369/bike-demand-monte-carlo

Verified on Python 3.13.5, NumPy 2.2.6, pandas 2.3.1. Data: UCI Bike Sharing, Hadi Fanaee-T, CC BY 4.0.

#Python #DataScience #MonteCarlo #OperationsAnalytics #MachineLearning
