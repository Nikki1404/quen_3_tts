import argparse
import asyncio
import csv
import json
import math
import statistics
import time
import wave
from datetime import datetime
from pathlib import Path

from websockets.asyncio.client import connect

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


def pct(values, p):
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    x = (len(values) - 1) * p / 100.0
    lo, hi = math.floor(x), math.ceil(x)
    return values[lo] if lo == hi else values[lo] + (values[hi] - values[lo]) * (x - lo)


def save_wav(path, pcm, sample_rate, channels, sample_width):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(channels)
        f.setsampwidth(sample_width)
        f.setframerate(sample_rate)
        f.writeframes(pcm)


async def one_request(req_no, concurrency, url, case, timeout, audio_dir):
    text, speaker, instruct = case
    result = {
        "concurrency": concurrency, "request_number": req_no, "success": False,
        "speaker": speaker, "text": text, "instruct": instruct,
        "instance_id": None, "ttfb_ms": None, "ttfa_ms": None,
        "total_ms": None, "connection_ms": None, "audio_file": None, "error": None,
    }

    conn_start = time.perf_counter()
    try:
        async with connect(url, max_size=None, open_timeout=30, ping_interval=20, ping_timeout=20) as ws:
            connected = time.perf_counter()
            result["connection_ms"] = (connected - conn_start) * 1000

            payload = {"text": text, "language": "English", "speaker": speaker, "instruct": instruct}
            start = time.perf_counter()
            await ws.send(json.dumps(payload, ensure_ascii=False))

            first_response = None
            first_audio = None
            sample_rate = None
            channels = 1
            sample_width = 2
            pcm = bytearray()

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

                typ = event.get("type")
                if typ == "accepted":
                    result["instance_id"] = event.get("instance_id")
                elif typ == "audio_start":
                    sample_rate = int(event["sample_rate"])
                    channels = int(event.get("channels", 1))
                    sample_width = int(event.get("sample_width", 2))
                elif typ == "done":
                    break
                elif typ == "error":
                    raise RuntimeError(event.get("error", "server error"))

            end = time.perf_counter()
            result["ttfb_ms"] = (first_response - start) * 1000 if first_response else None
            result["ttfa_ms"] = (first_audio - start) * 1000 if first_audio else None
            result["total_ms"] = (end - start) * 1000

            if not pcm:
                raise RuntimeError("No audio bytes received")
            if sample_rate is None:
                raise RuntimeError("No sample rate received")

            inst = (result["instance_id"] or "unknown")[:8]
            filename = f"c{concurrency:03d}_req{req_no:03d}_{speaker}_inst-{inst}.wav"
            audio_path = audio_dir / filename
            save_wav(audio_path, bytes(pcm), sample_rate, channels, sample_width)
            result["audio_file"] = str(audio_path)
            result["success"] = True

    except Exception as e:
        result["total_ms"] = (time.perf_counter() - conn_start) * 1000
        result["error"] = str(e)

    return result


async def run_level(concurrency, url, timeout, audio_dir):
    print("\n" + "=" * 100)
    print(f"CONCURRENCY = {concurrency}")
    print("=" * 100)

    started = time.perf_counter()
    tasks = [
        one_request(i + 1, concurrency, url, TEST_CASES[i % len(TEST_CASES)], timeout, audio_dir)
        for i in range(concurrency)
    ]
    results = await asyncio.gather(*tasks)
    wall = time.perf_counter() - started

    ok = [r for r in results if r["success"]]
    fail = [r for r in results if not r["success"]]
    instance_ids = sorted({r["instance_id"] for r in ok if r.get("instance_id")})

    for r in results:
        if r["success"]:
            print(
                f"req={r['request_number']:02d} OK "
                f"inst={(r['instance_id'] or '-')[:8]} "
                f"ttfb={r['ttfb_ms']:.1f}ms "
                f"ttfa={r['ttfa_ms']:.1f}ms "
                f"total={r['total_ms']:.1f}ms "
                f"audio={Path(r['audio_file']).name}"
            )
        else:
            print(f"req={r['request_number']:02d} FAIL error={r['error']}")

    ttfb = [r["ttfb_ms"] for r in ok if r["ttfb_ms"] is not None]
    ttfa = [r["ttfa_ms"] for r in ok if r["ttfa_ms"] is not None]
    totals = [r["total_ms"] for r in ok if r["total_ms"] is not None]

    summary = {
        "concurrency": concurrency,
        "successful": len(ok),
        "failed": len(fail),
        "success_rate_pct": len(ok) / concurrency * 100,
        "instances_observed": len(instance_ids) if instance_ids else None,
        "instance_ids": instance_ids,
        "wall_time_s": wall,
        "ttfb_avg_ms": statistics.mean(ttfb) if ttfb else None,
        "ttfb_p50_ms": pct(ttfb, 50),
        "ttfb_p95_ms": pct(ttfb, 95),
        "ttfa_avg_ms": statistics.mean(ttfa) if ttfa else None,
        "ttfa_p50_ms": pct(ttfa, 50),
        "ttfa_p95_ms": pct(ttfa, 95),
        "total_avg_ms": statistics.mean(totals) if totals else None,
        "total_p50_ms": pct(totals, 50),
        "total_p95_ms": pct(totals, 95),
    }

    print("-" * 100)
    print(f"Success           : {len(ok)}/{concurrency}")
    print(f"Failures          : {len(fail)}")
    print(f"Instances observed: {summary['instances_observed']}")
    print(f"TTFA p50          : {summary['ttfa_p50_ms']:.2f} ms" if summary["ttfa_p50_ms"] is not None else "TTFA p50          : -")
    print(f"TTFA p95          : {summary['ttfa_p95_ms']:.2f} ms" if summary["ttfa_p95_ms"] is not None else "TTFA p95          : -")
    print(f"Total p95         : {summary['total_p95_ms']:.2f} ms" if summary["total_p95_ms"] is not None else "Total p95         : -")
    print(f"Wall time         : {wall:.2f} s")
    print("-" * 100)

    return summary, results


def save_csv(path, rows):
    if not rows:
        return
    fields = [
        "concurrency", "request_number", "success", "speaker", "text", "instruct",
        "instance_id", "connection_ms", "ttfb_ms", "ttfa_ms", "total_ms",
        "audio_file", "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


async def main(args):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(args.output_dir) / f"qwen_loadtest_{stamp}"
    audio_dir = root / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    requests = []

    last_full_success = None
    first_failure = None
    first_two_instances = None

    print(f"Endpoint   : {args.url}")
    print(f"Levels     : {args.levels}")
    print(f"Text count : {len(TEST_CASES)}")
    print(f"Output     : {root}")

    for idx, concurrency in enumerate(args.levels):
        s, rows = await run_level(concurrency, args.url, args.timeout, audio_dir)
        summaries.append(s)
        requests.extend(rows)

        if s["failed"] == 0:
            last_full_success = concurrency
        elif first_failure is None:
            first_failure = concurrency

        if s["instances_observed"] and s["instances_observed"] >= 2 and first_two_instances is None:
            first_two_instances = concurrency

        (root / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        (root / "requests.json").write_text(json.dumps(requests, indent=2), encoding="utf-8")
        save_csv(root / "requests.csv", requests)

        if idx < len(args.levels) - 1:
            print(f"Waiting {args.pause}s...")
            await asyncio.sleep(args.pause)

    print("\n" + "#" * 100)
    print("FINAL SUMMARY")
    print("#" * 100)
    print(f"{'CONC':<8}{'SUCCESS':<12}{'FAIL':<8}{'INST':<8}{'TTFA-P50':<14}{'TTFA-P95':<14}{'TOTAL-P95':<14}")
    print("-" * 80)

    for s in summaries:
        print(
            f"{s['concurrency']:<8}"
            f"{str(s['successful']) + '/' + str(s['concurrency']):<12}"
            f"{s['failed']:<8}"
            f"{str(s['instances_observed'] or '-'): <8}"
            f"{(format(s['ttfa_p50_ms'], '.1f') if s['ttfa_p50_ms'] is not None else '-'): <14}"
            f"{(format(s['ttfa_p95_ms'], '.1f') if s['ttfa_p95_ms'] is not None else '-'): <14}"
            f"{(format(s['total_p95_ms'], '.1f') if s['total_p95_ms'] is not None else '-'): <14}"
        )

    print("\n" + "=" * 100)
    print(f"Last fully successful concurrency : {last_full_success}")
    print(f"First failing concurrency         : {first_failure}")
    print(f"First level with 2 instances      : {first_two_instances}")
    print(f"Audio folder                      : {audio_dir}")
    print(f"CSV                               : {root / 'requests.csv'}")
    print(f"JSON                              : {root / 'requests.json'}")
    print("=" * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=[1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80],
    )
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--pause", type=int, default=20)
    parser.add_argument("--output-dir", default="qwen_loadtest_results")
    args = parser.parse_args()
    asyncio.run(main(args))
