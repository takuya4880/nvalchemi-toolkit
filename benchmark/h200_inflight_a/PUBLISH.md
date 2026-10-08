# Publish this branch only

The local branch is `benchmark/h200-inflight-a`. The public remote `origin` is
`https://github.com/YuyaoHuang330/nvalchemi-toolkit.git`. No commit, push or PR
has been made by the preparation process.

From the checkout, commit the staged preparation and push this branch:

```bash
cd /scratch/users/ntu/yuyao003/Hackthon/nvalchemi-toolkit
git commit -s -m "docs: add H200 conventional inflight benchmark"
git push -u origin benchmark/h200-inflight-a
```

Then open a **Draft PR** with these settings:

- Base repository: `takuya4880/nvalchemi-toolkit`
- Base branch: `feat/inflight-symmetry-relaxation`
- Head repository: `YuyaoHuang330/nvalchemi-toolkit`
- Compare branch: `benchmark/h200-inflight-a`

Use `PR_DESCRIPTION.md` for the title/body. Check that the file diff contains
only this benchmark directory and the explicitly documented CI alignment.
Do not target NVIDIA/main or push all local branches: other local branches
contain private development and are outside this delivery.

This public A-only package includes historical results, not new GPU measurements.
Review `VALIDATION.json` and the README before presenting it as a reproduction.
