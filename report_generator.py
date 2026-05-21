"""
report_generator.py
Generates professional HTML loan decision reports.
Approved: repayment schedule, warnings, what you stand to lose.
Declined: reasons, improvement steps, re-application guide.
"""

import math
from datetime import datetime, timedelta


def amortisation_schedule(principal: float, annual_rate: float, months: int):
    r = annual_rate / 100 / 12
    if r == 0:
        monthly = principal / months
    else:
        monthly = principal * r * (1 + r) ** months / ((1 + r) ** months - 1)

    rows = []
    balance = principal
    for m in range(1, months + 1):
        interest = balance * r
        principal_payment = monthly - interest
        balance -= principal_payment
        rows.append({
            "month": m,
            "payment": monthly,
            "principal": principal_payment,
            "interest": interest,
            "balance": max(0, balance),
        })
    total_paid = monthly * months
    total_interest = total_paid - principal
    return rows, monthly, total_paid, total_interest


def generate_approved_report(applicant_name: str, loan_amount: float, interest_rate: float,
                              loan_term_months: int, score: int, risk_tier: str,
                              top_factors: list, approval_date: str = None) -> str:
    if not approval_date:
        approval_date = datetime.today().strftime("%d %B %Y")

    schedule, monthly_payment, total_paid, total_interest = amortisation_schedule(
        loan_amount, interest_rate, loan_term_months
    )

    start_date = datetime.today()
    schedule_rows = ""
    for row in schedule:
        pay_date = start_date + timedelta(days=30 * row["month"])
        schedule_rows += f"""
        <tr>
            <td>{row['month']}</td>
            <td>{pay_date.strftime('%d %b %Y')}</td>
            <td>R{row['payment']:,.2f}</td>
            <td>R{row['principal']:,.2f}</td>
            <td>R{row['interest']:,.2f}</td>
            <td>R{row['balance']:,.2f}</td>
        </tr>"""

    factors_html = ""
    for f in top_factors[:5]:
        icon = "✅" if f.get("positive", True) else "⚠️"
        factors_html += f"""
        <div class="factor-row">
            <span class="factor-icon">{icon}</span>
            <div class="factor-text">
                <strong>{f['name']}</strong>
                <span>{f['detail']}</span>
            </div>
        </div>"""

    danger_monthly = monthly_payment * 1.10
    penalty_example = monthly_payment * 0.05

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Loan Approval Report — {applicant_name}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'DM Sans', sans-serif; background: #F4F6F9; color: #1A1A2E; }}

  .page {{ max-width: 820px; margin: 0 auto; background: white; }}

  /* HEADER */
  .header {{ background: linear-gradient(135deg, #0A1628 0%, #007B8A 100%); padding: 48px 48px 40px; position: relative; overflow: hidden; }}
  .header::before {{ content: ''; position: absolute; top: -60px; right: -60px; width: 200px; height: 200px; border-radius: 50%; background: rgba(201,168,76,0.15); }}
  .header-badge {{ display: inline-block; background: #1A6B3C; color: white; font-size: 11px; font-weight: 600; letter-spacing: 3px; text-transform: uppercase; padding: 6px 16px; border-radius: 20px; margin-bottom: 20px; }}
  .header-title {{ font-family: 'DM Serif Display', serif; color: white; font-size: 36px; line-height: 1.2; margin-bottom: 8px; }}
  .header-sub {{ color: rgba(232,244,246,0.80); font-size: 14px; font-weight: 300; }}
  .header-meta {{ margin-top: 28px; display: flex; gap: 40px; flex-wrap: wrap; }}
  .header-meta-item label {{ color: #C9A84C; font-size: 10px; letter-spacing: 2px; text-transform: uppercase; display: block; margin-bottom: 4px; }}
  .header-meta-item span {{ color: white; font-size: 15px; font-weight: 500; }}

  /* SCORE BAND */
  .score-band {{ background: #E8F4F6; padding: 28px 48px; display: flex; align-items: center; gap: 40px; border-bottom: 3px solid #007B8A; }}
  .score-circle {{ width: 90px; height: 90px; border-radius: 50%; background: #1A6B3C; display: flex; flex-direction: column; align-items: center; justify-content: center; flex-shrink: 0; box-shadow: 0 4px 20px rgba(26,107,60,0.3); }}
  .score-circle .score-num {{ color: white; font-size: 28px; font-weight: 700; line-height: 1; }}
  .score-circle .score-label {{ color: rgba(255,255,255,0.80); font-size: 9px; letter-spacing: 1px; text-transform: uppercase; }}
  .score-info h3 {{ color: #0A1628; font-size: 18px; font-weight: 600; margin-bottom: 6px; }}
  .score-info p {{ color: #555; font-size: 13px; line-height: 1.6; }}

  /* SECTIONS */
  .section {{ padding: 36px 48px; border-bottom: 1px solid #EEEEEE; }}
  .section-title {{ font-family: 'DM Serif Display', serif; font-size: 22px; color: #0A1628; margin-bottom: 20px; display: flex; align-items: center; gap: 10px; }}
  .section-title .dot {{ width: 8px; height: 8px; border-radius: 50%; background: #C9A84C; flex-shrink: 0; }}

  /* LOAN SUMMARY GRID */
  .loan-grid {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }}
  .loan-card {{ background: #F8FAFB; border: 1px solid #E8E8E8; border-radius: 10px; padding: 18px; text-align: center; }}
  .loan-card label {{ display: block; font-size: 10px; color: #888; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 8px; }}
  .loan-card .amount {{ font-size: 22px; font-weight: 700; color: #0A1628; }}
  .loan-card .amount.highlight {{ color: #007B8A; }}

  /* FACTORS */
  .factor-row {{ display: flex; align-items: flex-start; gap: 14px; padding: 12px 0; border-bottom: 1px solid #F0F0F0; }}
  .factor-row:last-child {{ border-bottom: none; }}
  .factor-icon {{ font-size: 18px; flex-shrink: 0; margin-top: 2px; }}
  .factor-text strong {{ display: block; font-size: 13px; color: #0A1628; margin-bottom: 2px; }}
  .factor-text span {{ font-size: 12px; color: #666; }}

  /* SCHEDULE TABLE */
  .schedule-table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  .schedule-table th {{ background: #0A1628; color: #C9A84C; padding: 10px 12px; text-align: left; font-weight: 600; letter-spacing: 0.5px; font-size: 11px; }}
  .schedule-table td {{ padding: 9px 12px; border-bottom: 1px solid #F0F0F0; }}
  .schedule-table tr:nth-child(even) td {{ background: #FAFAFA; }}
  .schedule-table tr:hover td {{ background: #E8F4F6; }}
  .schedule-totals {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin-top: 16px; }}
  .total-box {{ background: #0A1628; border-radius: 8px; padding: 14px 16px; text-align: center; }}
  .total-box label {{ color: #C9A84C; font-size: 9px; letter-spacing: 1.5px; text-transform: uppercase; display: block; margin-bottom: 6px; }}
  .total-box span {{ color: white; font-size: 16px; font-weight: 700; }}

  /* DANGER SECTION */
  .danger-section {{ background: linear-gradient(135deg, #2D0000, #5C0000); padding: 36px 48px; }}
  .danger-title {{ font-family: 'DM Serif Display', serif; color: #FF6B6B; font-size: 22px; margin-bottom: 8px; }}
  .danger-subtitle {{ color: rgba(255,255,255,0.70); font-size: 13px; margin-bottom: 24px; }}
  .danger-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  .danger-card {{ background: rgba(255,255,255,0.08); border: 1px solid rgba(255,100,100,0.30); border-radius: 10px; padding: 18px; }}
  .danger-card .danger-icon {{ font-size: 24px; margin-bottom: 10px; }}
  .danger-card h4 {{ color: #FF9999; font-size: 13px; font-weight: 600; margin-bottom: 8px; }}
  .danger-card p {{ color: rgba(255,255,255,0.75); font-size: 12px; line-height: 1.6; }}
  .danger-card .danger-amount {{ color: #FF6B6B; font-size: 20px; font-weight: 700; margin: 8px 0 4px; }}

  /* RESPONSIBLE LENDING */
  .responsible {{ background: #F0FFF4; border-left: 4px solid #1A6B3C; padding: 20px 24px; border-radius: 0 8px 8px 0; margin-top: 20px; }}
  .responsible h4 {{ color: #1A6B3C; font-size: 13px; font-weight: 600; margin-bottom: 8px; }}
  .responsible p {{ color: #333; font-size: 12px; line-height: 1.7; }}

  /* FOOTER */
  .footer {{ background: #0A1628; padding: 28px 48px; text-align: center; }}
  .footer p {{ color: rgba(232,244,246,0.50); font-size: 11px; line-height: 1.8; }}

  @media print {{ body {{ background: white; }} .page {{ max-width: 100%; }} }}
</style>
</head>
<body>
<div class="page">

  <!-- HEADER -->
  <div class="header">
    <div class="header-badge">✓ Application Approved</div>
    <div class="header-title">Congratulations,<br>{applicant_name}.</div>
    <div class="header-sub">Your loan application has been reviewed and approved by our credit model.</div>
    <div class="header-meta">
      <div class="header-meta-item">
        <label>Approval Date</label>
        <span>{approval_date}</span>
      </div>
      <div class="header-meta-item">
        <label>Reference</label>
        <span>DQ-{score}-2026</span>
      </div>
      <div class="header-meta-item">
        <label>Risk Classification</label>
        <span>{risk_tier}</span>
      </div>
    </div>
  </div>

  <!-- SCORE BAND -->
  <div class="score-band">
    <div class="score-circle">
      <span class="score-num">{score}</span>
      <span class="score-label">Score</span>
    </div>
    <div class="score-info">
      <h3>Credit Score: {score} / 900 — {risk_tier}</h3>
      <p>Your credit score places you in the <strong>{risk_tier}</strong> category.
      Scores above 650 qualify for standard lending terms. Your profile was assessed
      across {len(top_factors)} key risk factors including income stability, debt obligations,
      credit utilisation, and repayment history.</p>
    </div>
  </div>

  <!-- LOAN SUMMARY -->
  <div class="section">
    <div class="section-title"><div class="dot"></div>Loan Summary</div>
    <div class="loan-grid">
      <div class="loan-card">
        <label>Loan Amount</label>
        <div class="amount">R{loan_amount:,.0f}</div>
      </div>
      <div class="loan-card">
        <label>Monthly Payment</label>
        <div class="amount highlight">R{monthly_payment:,.2f}</div>
      </div>
      <div class="loan-card">
        <label>Total Repayable</label>
        <div class="amount">R{total_paid:,.0f}</div>
      </div>
      <div class="loan-card">
        <label>Interest Rate (p.a.)</label>
        <div class="amount">{interest_rate:.1f}%</div>
      </div>
      <div class="loan-card">
        <label>Term</label>
        <div class="amount">{loan_term_months} months</div>
      </div>
      <div class="loan-card">
        <label>Total Interest</label>
        <div class="amount">R{total_interest:,.0f}</div>
      </div>
    </div>
  </div>

  <!-- APPROVAL FACTORS -->
  <div class="section">
    <div class="section-title"><div class="dot"></div>Why You Were Approved</div>
    {factors_html}
  </div>

  <!-- REPAYMENT SCHEDULE -->
  <div class="section">
    <div class="section-title"><div class="dot"></div>Repayment Schedule</div>
    <div style="overflow-x: auto;">
    <table class="schedule-table">
      <thead>
        <tr>
          <th>Month</th><th>Due Date</th><th>Payment</th>
          <th>Principal</th><th>Interest</th><th>Balance</th>
        </tr>
      </thead>
      <tbody>{schedule_rows}</tbody>
    </table>
    </div>
    <div class="schedule-totals">
      <div class="total-box">
        <label>Total Payments</label>
        <span>{loan_term_months}</span>
      </div>
      <div class="total-box">
        <label>Total Paid</label>
        <span>R{total_paid:,.0f}</span>
      </div>
      <div class="total-box">
        <label>Total Interest Cost</label>
        <span>R{total_interest:,.0f}</span>
      </div>
    </div>
  </div>

  <!-- DANGER SECTION -->
  <div class="danger-section">
    <div class="danger-title">⚠ What You Stand to Lose If You Don't Pay</div>
    <div class="danger-subtitle">This section is not here to scare you. It is here to protect you. Please read it carefully.</div>
    <div class="danger-grid">
      <div class="danger-card">
        <div class="danger-icon">💳</div>
        <h4>Credit Bureau Listing</h4>
        <p>Missing a payment by more than 20 days results in a negative listing on the
        National Credit Bureau. This stays on your record for <strong>5 years</strong>
        and will affect every future loan, rental, or credit application.</p>
      </div>
      <div class="danger-card">
        <div class="danger-icon">📈</div>
        <h4>Penalty Interest & Fees</h4>
        <p>A missed payment triggers a late fee and compound interest on the outstanding balance.
        Your effective monthly payment jumps to approximately:</p>
        <div class="danger-amount">R{danger_monthly:,.2f}</div>
        <p>per month until arrears are cleared.</p>
      </div>
      <div class="danger-card">
        <div class="danger-icon">⚖️</div>
        <h4>Legal Action</h4>
        <p>After 3 consecutive missed payments, the lender may hand the account to a
        debt collection agency or initiate <strong>Section 129 legal proceedings</strong>
        under the National Credit Act — resulting in a court order and potential asset attachment.</p>
      </div>
      <div class="danger-card">
        <div class="danger-icon">🏠</div>
        <h4>Asset Risk & Future Opportunities</h4>
        <p>Default affects your ability to rent property, qualify for a mortgage, or
        access any further credit for up to <strong>5 years</strong>. Employers in
        financial services may also conduct credit checks during hiring.</p>
      </div>
    </div>
    <div class="responsible">
      <h4>✅ Responsible Lending Commitment</h4>
      <p>If you are struggling to make a payment, contact us <strong>before</strong> you miss it.
      The National Credit Act gives you the right to apply for <strong>debt review</strong> or
      negotiate a <strong>payment holiday</strong>. Acting early protects your credit record.
      Ignoring the problem makes it significantly worse. You can reach the National Debt Helpline
      at <strong>0861 17 28 82</strong> — free, confidential, and independent.</p>
    </div>
  </div>

  <div class="footer">
    <p>This report was generated by the FNB DataQuest 2026 Credit Intelligence Platform.<br>
    Moloi Qolani Truelove & Tshegofatso Tshepang Chikwane — Sol Plaatje University<br>
    For illustrative and educational purposes only. Not a legally binding credit agreement.</p>
  </div>

</div>
</body>
</html>"""


def generate_declined_report(applicant_name: str, score: int, risk_tier: str,
                              top_negative_factors: list, improvement_steps: list,
                              decline_date: str = None) -> str:
    if not decline_date:
        decline_date = datetime.today().strftime("%d %B %Y")

    reasons_html = ""
    for i, f in enumerate(top_negative_factors[:3], 1):
        reasons_html += f"""
        <div class="reason-card">
          <div class="reason-num">{i:02d}</div>
          <div class="reason-body">
            <h4>{f['name']}</h4>
            <p>{f['detail']}</p>
            <div class="impact-bar">
              <div class="impact-fill" style="width:{min(f.get('impact_pct', 50), 100)}%"></div>
            </div>
            <span class="impact-label">Risk contribution: {f.get('impact_pct', 50):.0f}%</span>
          </div>
        </div>"""

    steps_html = ""
    for step in improvement_steps:
        steps_html += f"""
        <div class="step-card">
          <div class="step-icon">{step['icon']}</div>
          <div class="step-body">
            <h4>{step['title']}</h4>
            <p>{step['detail']}</p>
            <div class="step-impact">Estimated score improvement: <strong>+{step.get('score_gain', 20)-step.get('score_gain', 20)//2}-{step.get('score_gain', 20)} points</strong></div>
          </div>
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Loan Decision Report — {applicant_name}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'DM Sans', sans-serif; background: #F4F6F9; color: #1A1A2E; }}
  .page {{ max-width: 820px; margin: 0 auto; background: white; }}

  .header {{ background: linear-gradient(135deg, #0A1628 0%, #1a0a0a 100%); padding: 48px; position: relative; overflow: hidden; }}
  .header::before {{ content: ''; position: absolute; top: -60px; right: -60px; width: 200px; height: 200px; border-radius: 50%; background: rgba(192,57,43,0.15); }}
  .header-badge {{ display: inline-block; background: rgba(192,57,43,0.85); color: white; font-size: 11px; font-weight: 600; letter-spacing: 3px; text-transform: uppercase; padding: 6px 16px; border-radius: 20px; margin-bottom: 20px; }}
  .header-title {{ font-family: 'DM Serif Display', serif; color: white; font-size: 34px; line-height: 1.25; margin-bottom: 8px; }}
  .header-sub {{ color: rgba(232,244,246,0.75); font-size: 14px; font-weight: 300; line-height: 1.6; }}
  .header-meta {{ margin-top: 28px; display: flex; gap: 40px; flex-wrap: wrap; }}
  .header-meta-item label {{ color: #C9A84C; font-size: 10px; letter-spacing: 2px; text-transform: uppercase; display: block; margin-bottom: 4px; }}
  .header-meta-item span {{ color: white; font-size: 14px; font-weight: 500; }}

  .score-band {{ background: #FDF0EE; padding: 28px 48px; display: flex; align-items: center; gap: 40px; border-bottom: 3px solid #C0392B; }}
  .score-circle {{ width: 90px; height: 90px; border-radius: 50%; background: #C0392B; display: flex; flex-direction: column; align-items: center; justify-content: center; flex-shrink: 0; box-shadow: 0 4px 20px rgba(192,57,43,0.3); }}
  .score-circle .score-num {{ color: white; font-size: 28px; font-weight: 700; line-height: 1; }}
  .score-circle .score-label {{ color: rgba(255,255,255,0.80); font-size: 9px; letter-spacing: 1px; text-transform: uppercase; }}
  .score-info h3 {{ color: #0A1628; font-size: 18px; font-weight: 600; margin-bottom: 6px; }}
  .score-info p {{ color: #555; font-size: 13px; line-height: 1.6; }}

  .section {{ padding: 36px 48px; border-bottom: 1px solid #EEEEEE; }}
  .section-title {{ font-family: 'DM Serif Display', serif; font-size: 22px; color: #0A1628; margin-bottom: 20px; display: flex; align-items: center; gap: 10px; }}
  .section-title .dot {{ width: 8px; height: 8px; border-radius: 50%; background: #C9A84C; flex-shrink: 0; }}

  .reason-card {{ display: flex; gap: 18px; padding: 18px 0; border-bottom: 1px solid #F0F0F0; }}
  .reason-card:last-child {{ border-bottom: none; }}
  .reason-num {{ width: 40px; height: 40px; border-radius: 8px; background: #C0392B; color: white; font-size: 16px; font-weight: 700; display: flex; align-items: center; justify-content: center; flex-shrink: 0; }}
  .reason-body h4 {{ font-size: 14px; color: #0A1628; font-weight: 600; margin-bottom: 6px; }}
  .reason-body p {{ font-size: 12px; color: #555; line-height: 1.6; margin-bottom: 10px; }}
  .impact-bar {{ height: 6px; background: #F0F0F0; border-radius: 3px; margin-bottom: 4px; }}
  .impact-fill {{ height: 100%; background: linear-gradient(90deg, #C9A84C, #C0392B); border-radius: 3px; }}
  .impact-label {{ font-size: 11px; color: #888; }}

  .step-card {{ display: flex; gap: 18px; padding: 18px; background: #F8FFF9; border: 1px solid #C8E6C9; border-radius: 10px; margin-bottom: 12px; }}
  .step-icon {{ font-size: 28px; flex-shrink: 0; }}
  .step-body h4 {{ font-size: 14px; color: #0A1628; font-weight: 600; margin-bottom: 6px; }}
  .step-body p {{ font-size: 12px; color: #444; line-height: 1.6; margin-bottom: 8px; }}
  .step-impact {{ font-size: 11px; color: #1A6B3C; font-weight: 500; }}

  .reapply-box {{ background: #E8F4F6; border-left: 4px solid #007B8A; border-radius: 0 10px 10px 0; padding: 24px; margin-top: 4px; }}
  .reapply-box h4 {{ color: #007B8A; font-size: 14px; font-weight: 600; margin-bottom: 10px; }}
  .reapply-box p {{ color: #333; font-size: 13px; line-height: 1.7; }}
  .reapply-box ul {{ margin-top: 10px; padding-left: 18px; }}
  .reapply-box li {{ font-size: 12px; color: #444; line-height: 2; }}

  .resources-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 4px; }}
  .resource-card {{ background: #F8FAFB; border: 1px solid #E0E0E0; border-radius: 8px; padding: 16px; }}
  .resource-card h5 {{ font-size: 12px; color: #0A1628; font-weight: 600; margin-bottom: 4px; }}
  .resource-card p {{ font-size: 11px; color: #666; line-height: 1.5; }}
  .resource-card a {{ color: #007B8A; font-weight: 500; }}

  .footer {{ background: #0A1628; padding: 28px 48px; text-align: center; }}
  .footer p {{ color: rgba(232,244,246,0.50); font-size: 11px; line-height: 1.8; }}
</style>
</head>
<body>
<div class="page">

  <div class="header">
    <div class="header-badge">✗ Application Outcome: Declined</div>
    <div class="header-title">Dear {applicant_name},<br>We were unable to approve<br>your application at this time.</div>
    <div class="header-sub">This decision is not permanent. This report explains exactly why,
    what you can do about it, and when you can reapply. Please read it fully — it is designed to help you.</div>
    <div class="header-meta">
      <div class="header-meta-item">
        <label>Decision Date</label>
        <span>{decline_date}</span>
      </div>
      <div class="header-meta-item">
        <label>Credit Score</label>
        <span>{score} / 900</span>
      </div>
      <div class="header-meta-item">
        <label>Risk Classification</label>
        <span>{risk_tier}</span>
      </div>
    </div>
  </div>

  <div class="score-band">
    <div class="score-circle">
      <span class="score-num">{score}</span>
      <span class="score-label">Score</span>
    </div>
    <div class="score-info">
      <h3>Credit Score: {score} / 900 — {risk_tier}</h3>
      <p>Your current score of <strong>{score}</strong> falls below our minimum threshold of <strong>580</strong>.
      The good news: credit scores are not fixed. Every item identified below is something
      you can actively improve. Most applicants who follow the steps in this report requalify
      within 6-12 months.</p>
    </div>
  </div>

  <div class="section">
    <div class="section-title"><div class="dot"></div>Why Your Application Was Declined</div>
    {reasons_html}
  </div>

  <div class="section">
    <div class="section-title"><div class="dot"></div>Your Improvement Plan</div>
    <p style="font-size:13px; color:#555; margin-bottom:20px;">
      These are the three highest-impact actions you can take right now. Each one is
      specific, measurable, and within your control.
    </p>
    {steps_html}

    <div class="reapply-box">
      <h4>📅 When Can You Reapply?</h4>
      <p>You may reapply in <strong>6 months</strong> from the date of this decision, provided:</p>
      <ul>
        <li>Your debt-to-income ratio has improved by at least 5 percentage points</li>
        <li>You have no new delinquencies or missed payments in the interim</li>
        <li>Your credit utilisation is below 70% on all active revolving accounts</li>
        <li>You can provide updated proof of income if your employment situation has changed</li>
      </ul>
      <p style="margin-top:12px; font-size:12px; color:#555;">
        Early reapplication without addressing the factors above is unlikely to succeed
        and may result in additional hard inquiries that temporarily lower your score further.
      </p>
    </div>
  </div>

  <div class="section">
    <div class="section-title"><div class="dot"></div>Support & Resources</div>
    <p style="font-size:13px; color:#555; margin-bottom:16px;">
      You do not have to navigate this alone. These organisations can help you at no cost.
    </p>
    <div class="resources-grid">
      <div class="resource-card">
        <h5>National Debt Helpline</h5>
        <p>Free, confidential debt counselling and credit assessment.<br>
        <a href="tel:0861172882">0861 17 28 82</a></p>
      </div>
      <div class="resource-card">
        <h5>National Credit Regulator (NCR)</h5>
        <p>For complaints about unfair credit decisions or to understand your rights.<br>
        <a href="tel:0860627627">0860 NCR NCR</a></p>
      </div>
      <div class="resource-card">
        <h5>Credit Bureau Dispute</h5>
        <p>If you believe your credit record contains errors, you have the right to dispute them free of charge with any registered credit bureau.</p>
      </div>
      <div class="resource-card">
        <h5>Debt Review</h5>
        <p>If you are over-indebted, a registered debt counsellor can restructure all your repayments into one affordable payment while protecting you from legal action.</p>
      </div>
    </div>
  </div>

  <div class="footer">
    <p>This report was generated by the FNB DataQuest 2026 Credit Intelligence Platform.<br>
    Moloi Qolani Truelove & Tshegofatso Tshepang Chikwane — Sol Plaatje University<br>
    For illustrative and educational purposes only. Not a legally binding credit decision.</p>
  </div>

</div>
</body>
</html>"""
