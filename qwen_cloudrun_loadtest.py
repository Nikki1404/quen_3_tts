import argparse
import asyncio
import csv
import json
import math
import statistics
import time
import wave
from datetime import datetime, timezone, timedelta
from pathlib import Path

from websockets.asyncio.client import connect

try:
    from google.cloud import monitoring_v3
except ImportError:
    monitoring_v3 = None


TEST_CASES = [
    ("Hello, thank you for calling Inspira Financial. How can I help you today?", "Aiden", "Speak in a professional customer service tone."),
    ("I understand your concern. Let me check your account and help you resolve this issue.", "Ryan", "Speak calmly and empathetically."),
    ("That is great news! Your request has been successfully processed.", "Aiden", "Speak happily and with excitement."),
    ("I am sorry, but we are currently unable to process this transaction.", "Ryan", "Speak seriously and professionally."),
    ("Before I can help you withdraw money, I will need to verify your identity.", "Aiden", "Speak professionally and clearly."),
    ("I completely understand that this situation may be frustrating. Let me review the details carefully and see what options are available for you.", "Ryan", "Speak empathetically and reassuringly."),
    ("Thank you for confirming your information. Your identity has now been verified and I can continue helping you with your account request.", "Aiden", "Speak confidently and professionally."),
    ("I understand that you are upset about the delay. I will do my best to help you resolve the issue as quickly as possible.", "Ryan", "Speak calmly while addressing an angry customer."),
    ("Your withdrawal request has been received. It may take approximately two to three business days for the funds to appear in your bank account.", "Aiden", "Speak slowly, clearly, and professionally."),
    ("Hi, thank you for calling Inspira Financial. What can I help you with today? I would also like to withdraw money from my account. To help you with that, I'll need to verify your identity.", "Aiden", "Speak in a very professional tone."),
    ("Your account balance is currently five thousand four hundred and twenty dollars.", "Ryan", "Speak clearly and naturally."),
    ("I can help you update your mailing address. First, I need to verify a few details on your account.", "Aiden", "Speak professionally and politely."),
    ("Your transaction has been declined because the available balance is lower than the requested amount.", "Ryan", "Speak seriously but politely."),
    ("Please do not worry. Your account is secure and there are no unauthorized transactions showing at this time.", "Aiden", "Speak calmly and reassuringly."),
    ("Congratulations! Your account verification has been successfully completed.", "Ryan", "Speak happily and positively."),
    ("I am going to place you on a brief hold while I review the transaction history.", "Aiden", "Speak politely in a customer service tone."),
    ("The payment was submitted yesterday and is currently being processed by the receiving bank.", "Ryan", "Speak clearly and professionally."),
    ("I apologize for the inconvenience. I understand that you expected the payment to arrive sooner.", "Aiden", "Speak sincerely and empathetically."),
    ("For security purposes, please confirm the last four digits of your registered phone number.", "Ryan", "Speak clearly and firmly while remaining professional."),
    ("Your request has been escalated to our specialist team and they will review it as soon as possible.", "Aiden", "Speak confidently and reassuringly."),
    ("I can see why this would be concerning. Let me explain exactly what happened with the transaction.", "Ryan", "Speak empathetically and calmly."),
    ("The transfer was completed successfully and the confirmation number is available in your transaction history.", "Aiden", "Speak positively and professionally."),
    ("Unfortunately, I cannot make that change until we complete the required identity verification.", "Ryan", "Speak firmly but respectfully."),
    ("Thank you for your patience. I have reviewed your account and found the reason for the delay.", "Aiden", "Speak warmly and professionally."),
    ("Your account is currently active and there are no restrictions preventing you from making a withdrawal.", "Ryan", "Speak confidently and clearly."),
    ("I am sorry to hear that you are having trouble accessing your account. Let us go through the recovery process together.", "Aiden", "Speak empathetically and reassuringly."),
    ("Please confirm whether you recognize a transaction for one hundred and twenty five dollars made yesterday evening.", "Ryan", "Speak seriously and carefully."),
    ("Everything looks good now. Your request has been approved and no further action is required.", "Aiden", "Speak happily and confidently."),
    ("I understand you would like this resolved immediately. I am checking the fastest available option for you now.", "Ryan", "Speak calmly and professionally while handling an impatient customer."),
    ("Thank you for calling Inspira Financial today. Is there anything else I can help you with before we end the call?", "Aiden", "Speak warmly, professionally, and politely."),
]


def utc_now():
    return datetime.now(timezone.utc)


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    x = (len(values) - 1) * p / 100.0
    lo, hi = math.floor(x), math.ceil(x)
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (x - lo)


def save_wav(path, pcm, sample_rate, channels, sample_width):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(channels)
        f.setsampwidth(sample_width)
        f.setframerate(sample_rate)
        f.writeframes(pcm)


async def one_request(req_no, batch_no, concurrency, url, case, timeout, audio_dir):
    text, speaker, instruct = case
    result = {
        "concurrency": concurrency,
        "batch_number": batch_no,
        "request_number": req_no,
        "success": False,
        "speaker": speaker,
        "text": text,
        "instruct": instruct,
        "ttfb_ms": None,
        "ttfa_ms": None,
        "total_ms": None,
        "connection_ms": None,
        "audio_bytes": 0,
        "audio_file": None,
        "error": None,
    }

    conn_start = time.perf_counter()

    try:
        async with connect(
            url,
            max_size=None,
            open_timeout=30,
            ping_interval=20,
            ping_timeout=20,
        ) as ws:
            connected = time.perf_counter()
            result["connection_ms"] = (connected - conn_start) * 1000

            payload = {
                "text": text,
                "language": "English",
                "speaker": speaker,
                "instruct": instruct,
            }

            start = time.perf_counter()
            await ws.send(json.dumps(payload, ensure_ascii=False))

            first_response = None
            first_audio = None
            pcm = bytearray()
            sample_rate = None
            channels = 1
            sample_width = 2

            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=timeout)
                now = time.perf_counter()

                if isinstance(msg, bytes):
                    if first_audio is None:
                        first_audio = now
                    pcm.extend(msg)
                    continue

                event = json.loads(msg)

                if first_response is None:
                    first_response = now

                event_type = event.get("type")

                if event_type == "audio_start":
                    sample_rate = int(event["sample_rate"])
                    channels = int(event.get("channels", 1))
                    sample_width = int(event.get("sample_width", 2))
                elif event_type == "done":
                    break
                elif event_type == "error":
                    raise RuntimeError(event.get("error", "Server returned error"))

            end = time.perf_counter()

            result["ttfb_ms"] = (first_response - start) * 1000 if first_response else None
            result["ttfa_ms"] = (first_audio - start) * 1000 if first_audio else None
            result["total_ms"] = (end - start) * 1000
            result["audio_bytes"] = len(pcm)

            if not pcm:
                raise RuntimeError("No audio bytes received")
            if sample_rate is None:
                raise RuntimeError("Sample rate not received")

            filename = f"c{concurrency:03d}_b{batch_no:03d}_req{req_no:04d}_{speaker}.wav"
            audio_path = audio_dir / filename
            save_wav(audio_path, bytes(pcm), sample_rate, channels, sample_width)

            result["audio_file"] = str(audio_path)
            result["success"] = True

    except Exception as exc:
        result["total_ms"] = (time.perf_counter() - conn_start) * 1000
        result["error"] = str(exc)

    return result


async def run_level(concurrency, url, timeout, duration, audio_dir):
    print("\n" + "=" * 100)
    print(f"CONCURRENCY {concurrency} | sustain for at least {duration}s")
    print("=" * 100)

    level_start_utc = utc_now()
    level_start_perf = time.perf_counter()

    all_results = []
    batch_no = 0
    request_counter = 0

    while True:
        elapsed = time.perf_counter() - level_start_perf
        if batch_no > 0 and elapsed >= duration:
            break

        batch_no += 1
        tasks = []

        for _ in range(concurrency):
            request_counter += 1
            case = TEST_CASES[(request_counter - 1) % len(TEST_CASES)]
            tasks.append(
                one_request(
                    request_counter,
                    batch_no,
                    concurrency,
                    url,
                    case,
                    timeout,
                    audio_dir,
                )
            )

        batch_results = await asyncio.gather(*tasks)
        all_results.extend(batch_results)

        ok = sum(1 for r in batch_results if r["success"])
        print(
            f"batch={batch_no:03d} "
            f"success={ok}/{concurrency} "
            f"elapsed={time.perf_counter() - level_start_perf:.1f}s"
        )

    level_end_utc = utc_now()

    successful = [r for r in all_results if r["success"]]
    failed = [r for r in all_results if not r["success"]]

    ttfb = [r["ttfb_ms"] for r in successful if r["ttfb_ms"] is not None]
    ttfa = [r["ttfa_ms"] for r in successful if r["ttfa_ms"] is not None]
    totals = [r["total_ms"] for r in successful if r["total_ms"] is not None]

    summary = {
        "concurrency": concurrency,
        "level_start_utc": level_start_utc.isoformat(),
        "level_end_utc": level_end_utc.isoformat(),
        "requests_total": len(all_results),
        "successful": len(successful),
        "failed": len(failed),
        "success_rate_pct": (len(successful) / len(all_results) * 100) if all_results else 0,
        "ttfb_p50_ms": percentile(ttfb, 50),
        "ttfb_p95_ms": percentile(ttfb, 95),
        "ttfa_p50_ms": percentile(ttfa, 50),
        "ttfa_p95_ms": percentile(ttfa, 95),
        "total_p50_ms": percentile(totals, 50),
        "total_p95_ms": percentile(totals, 95),
        "monitoring_max_active_instances": None,
    }

    print(
        f"SUMMARY: success={len(successful)}/{len(all_results)} "
        f"ttfa_p50={summary['ttfa_p50_ms']} "
        f"ttfa_p95={summary['ttfa_p95_ms']}"
    )

    return summary, all_results


def query_max_active_instances(project, service, region, start_time, end_time):
    if monitoring_v3 is None:
        raise RuntimeError(
            "google-cloud-monitoring is missing. "
            "Install with: pip install google-cloud-monitoring"
        )

    client = monitoring_v3.MetricServiceClient()

    metric_filter = (
        'metric.type="run.googleapis.com/container/instance_count" '
        'AND resource.type="cloud_run_revision" '
        f'AND resource.labels.service_name="{service}" '
        f'AND resource.labels.location="{region}" '
        'AND metric.labels.state="active"'
    )

    interval = monitoring_v3.TimeInterval(
        {
            "start_time": start_time - timedelta(seconds=60),
            "end_time": end_time + timedelta(seconds=60),
        }
    )

    aggregation = monitoring_v3.Aggregation(
        {
            "alignment_period": {"seconds": 60},
            "per_series_aligner": monitoring_v3.Aggregation.Aligner.ALIGN_MAX,
            "cross_series_reducer": monitoring_v3.Aggregation.Reducer.REDUCE_SUM,
        }
    )

    series = client.list_time_series(
        request={
            "name": f"projects/{project}",
            "filter": metric_filter,
            "interval": interval,
            "view": monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
            "aggregation": aggregation,
        }
    )

    values = []
    points = []

    for ts in series:
        for point in ts.points:
            value = int(point.value.int64_value)
            values.append(value)
            points.append(
                {
                    "time": point.interval.end_time.isoformat(),
                    "instances": value,
                }
            )

    return (max(values) if values else None), sorted(points, key=lambda x: x["time"])


def save_csv(path, rows):
    fields = [
        "concurrency",
        "batch_number",
        "request_number",
        "success",
        "speaker",
        "text",
        "instruct",
        "connection_ms",
        "ttfb_ms",
        "ttfa_ms",
        "total_ms",
        "audio_bytes",
        "audio_file",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


async def main(args):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(args.output_dir) / f"qwen_loadtest_{stamp}"
    audio_dir = root / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    requests = []

    print(f"URL       : {args.url}")
    print(f"Project   : {args.project}")
    print(f"Service   : {args.service}")
    print(f"Region    : {args.region}")
    print(f"Levels    : {args.levels}")
    print(f"Duration  : {args.duration}s/level")
    print(f"Outputs   : {root}")

    for idx, concurrency in enumerate(args.levels):
        summary, level_requests = await run_level(
            concurrency,
            args.url,
            args.timeout,
            args.duration,
            audio_dir,
        )

        summaries.append(summary)
        requests.extend(level_requests)

        (root / "requests.json").write_text(json.dumps(requests, indent=2), encoding="utf-8")
        save_csv(root / "requests.csv", requests)

        if idx < len(args.levels) - 1:
            print(f"Cooldown {args.pause}s...")
            await asyncio.sleep(args.pause)

    print(
        f"\nWaiting {args.monitor_wait}s for Cloud Monitoring "
        "instance-count samples..."
    )
    await asyncio.sleep(args.monitor_wait)

    for summary in summaries:
        start_time = datetime.fromisoformat(summary["level_start_utc"])
        end_time = datetime.fromisoformat(summary["level_end_utc"])

        try:
            max_instances, points = query_max_active_instances(
                args.project,
                args.service,
                args.region,
                start_time,
                end_time,
            )
            summary["monitoring_max_active_instances"] = max_instances
            summary["monitoring_points"] = points
        except Exception as exc:
            summary["monitoring_error"] = str(exc)

    (root / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    last_full_success = None
    first_failure = None
    first_two_instances = None

    print("\n" + "#" * 100)
    print("FINAL SUMMARY")
    print("#" * 100)
    print(
        f"{'CONC':<8}{'SUCCESS':<14}{'FAIL':<8}{'INST':<8}"
        f"{'TTFA-P50':<14}{'TTFA-P95':<14}{'TOTAL-P95':<14}"
    )
    print("-" * 80)

    for s in summaries:
        if s["failed"] == 0:
            last_full_success = s["concurrency"]
        elif first_failure is None:
            first_failure = s["concurrency"]

        inst = s.get("monitoring_max_active_instances")

        if inst is not None and inst >= 2 and first_two_instances is None:
            first_two_instances = s["concurrency"]

        success_text = f"{s['successful']}/{s['requests_total']}"

        print(
            f"{s['concurrency']:<8}"
            f"{success_text:<14}"
            f"{s['failed']:<8}"
            f"{str(inst if inst is not None else '-'): <8}"
            f"{str(round(s['ttfa_p50_ms'], 1) if s['ttfa_p50_ms'] is not None else '-'): <14}"
            f"{str(round(s['ttfa_p95_ms'], 1) if s['ttfa_p95_ms'] is not None else '-'): <14}"
            f"{str(round(s['total_p95_ms'], 1) if s['total_p95_ms'] is not None else '-'): <14}"
        )

    print("\n" + "=" * 100)
    print(f"Last fully failure-free level      : {last_full_success}")
    print(f"First level containing a failure   : {first_failure}")
    print(f"First level where 2 instances seen : {first_two_instances}")
    print(f"Audio folder                       : {audio_dir}")
    print(f"Requests CSV                       : {root / 'requests.csv'}")
    print(f"Summary JSON                       : {root / 'summary.json'}")
    print("=" * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--service", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=[1, 5, 10, 15, 20, 22, 24, 25, 26, 28, 30, 35, 40, 45, 50, 60, 70, 80],
    )
    parser.add_argument("--duration", type=int, default=70)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--pause", type=int, default=30)
    parser.add_argument("--monitor-wait", type=int, default=130)
    parser.add_argument("--output-dir", default="qwen_loadtest_results")
    args = parser.parse_args()

    asyncio.run(main(args))
