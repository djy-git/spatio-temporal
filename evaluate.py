from ydj.util import *


if __name__ == '__main__':
    common_cols = ['TurbID', 'Day', 'Tmstamp']

    assert len(sys.argv) == 2, "RIGHT USAGE: 'python evaluate.py {submission_file_name}'"
    submission_df = pd.read_csv(join(PATH.output, sys.argv[1]))
    target_df     = pd.read_csv(PATH.target)[common_cols + ['Patv']].rename(columns={'Patv': 'Patv_target'})
    submission_target_df = pd.merge(left=submission_df, right=target_df, how='left', on=common_cols, sort=False)

    scores = pd.DataFrame(columns=['rmse', 'mae', 'mean'], index=pd.Index(submission_target_df['TurbID'].unique(), name='TurbID'))
    for turbid in scores.index:
        df = submission_target_df.query(f"TurbID == {turbid}")
        preds, targets = df['Patv'], df['Patv_target']
        scores.loc[turbid, ['rmse', 'mae']] = (np.sqrt(np.mean((preds - targets)**2)),
                                               np.mean(np.abs(preds - targets)))

    scores['mean'] = (scores['rmse'] + scores['mae'])/2
    print(f"Final score: {scores['mean'].mean():.2f}")
    print(tabulate(scores, headers='keys', tablefmt='psql'))
