#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
#
# This file is part of Encre.
# The Encre project belongs to the Dunimd Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# DISCLAIMER: Users must comply with applicable AI regulations.
# Non-compliance may result in service termination or legal liability.

"""Tests for the cron scheduler: :class:`CronSchedule`, :class:`ScheduledJob`,
:class:`EncreScheduler`, :class:`ScheduleType`, and :class:`JobState`.
"""

import json
import os
import tempfile
import time
from pathlib import Path

import pytest

from encre.scheduler import (
    CronSchedule,
    JobState,
    ScheduledJob,
    ScheduleType,
    EncreScheduler,
)


# ===========================================================================
# CronSchedule.parse()
# ===========================================================================

class TestCronScheduleParse:
    """Test :meth:`CronSchedule.parse` with valid and invalid expressions."""

    def test_parse_every_minute(self):
        cs = CronSchedule.parse("* * * * *")
        assert cs.minute == "*"
        assert cs.hour == "*"
        assert cs.day_of_month == "*"
        assert cs.month == "*"
        assert cs.day_of_week == "*"

    def test_parse_specific_time(self):
        cs = CronSchedule.parse("30 9 15 3 *")
        assert cs.minute == "30"
        assert cs.hour == "9"
        assert cs.day_of_month == "15"
        assert cs.month == "3"
        assert cs.day_of_week == "*"

    def test_parse_with_ranges(self):
        cs = CronSchedule.parse("0 9-17 * * 1-5")
        assert cs.hour == "9-17"
        assert cs.day_of_week == "1-5"

    def test_parse_with_step(self):
        cs = CronSchedule.parse("*/5 * * * *")
        assert cs.minute == "*/5"

    def test_parse_with_named_days(self):
        cs = CronSchedule.parse("0 9 * * mon")
        assert cs.day_of_week == "mon"

    def test_parse_with_comma_list(self):
        cs = CronSchedule.parse("0,30 9,17 * * *")
        assert cs.minute == "0,30"
        assert cs.hour == "9,17"

    def test_parse_too_few_fields_raises(self):
        with pytest.raises(ValueError, match="Expected 5 fields"):
            CronSchedule.parse("0 9 * *")

    def test_parse_too_many_fields_raises(self):
        with pytest.raises(ValueError, match="Expected 5 fields"):
            CronSchedule.parse("0 9 * * * *")

    def test_parse_empty_raises(self):
        with pytest.raises(ValueError, match="Expected 5 fields"):
            CronSchedule.parse("")

    def test_parse_whitespace_only_raises(self):
        with pytest.raises(ValueError):
            CronSchedule.parse("     ")

    def test_to_expression_roundtrip(self):
        expr = "*/10 8-18 1,15 * mon-fri"
        cs = CronSchedule.parse(expr)
        assert cs.to_expression() == expr


# ===========================================================================
# CronSchedule._match_field
# ===========================================================================

class TestMatchField:
    """Test :meth:`CronSchedule._match_field` logic."""

    def test_star_matches_all(self):
        cs = CronSchedule.parse("* * * * *")
        for v in range(60):
            assert cs._match_field(v, "*") is True

    def test_exact_match(self):
        cs = CronSchedule.parse("* * * * *")
        assert cs._match_field(30, "30") is True
        assert cs._match_field(31, "30") is False

    def test_step_match(self):
        cs = CronSchedule.parse("* * * * *")
        assert cs._match_field(0, "*/5") is True
        assert cs._match_field(5, "*/5") is True
        assert cs._match_field(7, "*/5") is False

    def test_step_with_base(self):
        cs = CronSchedule.parse("* * * * *")
        assert cs._match_field(10, "10/5") is True
        assert cs._match_field(15, "10/5") is True
        assert cs._match_field(9, "10/5") is False

    def test_range_match(self):
        cs = CronSchedule.parse("* * * * *")
        assert cs._match_field(10, "9-17") is True
        assert cs._match_field(9, "9-17") is True
        assert cs._match_field(17, "9-17") is True
        assert cs._match_field(8, "9-17") is False
        assert cs._match_field(18, "9-17") is False

    def test_comma_list_match(self):
        cs = CronSchedule.parse("* * * * *")
        assert cs._match_field(0, "0,30") is True
        assert cs._match_field(30, "0,30") is True
        assert cs._match_field(15, "0,30") is False

    def test_invalid_int_ignored(self):
        cs = CronSchedule.parse("* * * * *")
        # Non-integer should not raise, just not match
        assert cs._match_field(5, "mon") is False


# ===========================================================================
# CronSchedule._normalize_dow
# ===========================================================================

class TestNormalizeDOW:
    """Test named day-of-week conversion."""

    def test_sun_to_0(self):
        assert CronSchedule._normalize_dow("sun") == "0"

    def test_mon_to_1(self):
        assert CronSchedule._normalize_dow("mon") == "1"

    def test_tue_to_2(self):
        assert CronSchedule._normalize_dow("tue") == "2"

    def test_wed_to_3(self):
        assert CronSchedule._normalize_dow("wed") == "3"

    def test_thu_to_4(self):
        assert CronSchedule._normalize_dow("thu") == "4"

    def test_fri_to_5(self):
        assert CronSchedule._normalize_dow("fri") == "5"

    def test_sat_to_6(self):
        assert CronSchedule._normalize_dow("sat") == "6"

    def test_case_insensitive(self):
        assert CronSchedule._normalize_dow("MON") == "1"
        assert CronSchedule._normalize_dow("Fri") == "5"

    def test_mixed_range(self):
        # "mon-fri" → names replaced in order: mon→1, fri→5 → "1-5"
        result = CronSchedule._normalize_dow("mon-fri")
        assert result == "1-5"

    def test_numeric_unchanged(self):
        assert CronSchedule._normalize_dow("5") == "5"


# ===========================================================================
# CronSchedule._weekday_cron
# ===========================================================================

class TestWeekdayCron:
    """Test the weekday calculation helper."""

    def test_known_date_monday(self):
        # 2024-01-01 was a Monday
        wday = CronSchedule._weekday_cron(2024, 1, 1)
        assert wday == 1  # Monday = 1 in cron

    def test_known_date_sunday(self):
        # 2024-01-07 was a Sunday
        wday = CronSchedule._weekday_cron(2024, 1, 7)
        assert wday == 0  # Sunday = 0 in cron

    def test_known_date_friday(self):
        # 2026-05-15 was a Friday
        wday = CronSchedule._weekday_cron(2026, 5, 15)
        assert wday == 5  # Friday = 5 in cron


# ===========================================================================
# CronSchedule.next_fire()
# ===========================================================================

class TestNextFire:
    """Test :meth:`CronSchedule.next_fire` with various patterns."""

    def test_next_fire_every_minute(self):
        cs = CronSchedule.parse("* * * * *")
        now = time.time()
        nf = cs.next_fire(now)
        assert nf is not None
        assert nf > now
        assert nf - now <= 62

    def test_next_fire_specific_minute(self):
        cs = CronSchedule.parse("7 * * * *")
        now = time.time()
        nf = cs.next_fire(now)
        assert nf is not None
        assert nf > now
        t = time.localtime(nf)
        assert t.tm_min == 7

    def test_next_fire_daily_9am(self):
        cs = CronSchedule.parse("0 9 * * *")
        now = time.time()
        nf = cs.next_fire(now)
        assert nf is not None
        t = time.localtime(nf)
        assert t.tm_hour == 9
        assert t.tm_min == 0

    def test_next_fire_weekdays(self):
        cs = CronSchedule.parse("0 9 * * 1-5")
        now = time.time()
        nf = cs.next_fire(now)
        assert nf is not None
        t = time.localtime(nf)
        assert 0 <= t.tm_wday <= 4  # Python: 0=Mon, 4=Fri
        assert t.tm_hour == 9

    def test_next_fire_on_monday(self):
        cs = CronSchedule.parse("0 12 * * mon")
        now = time.time()
        nf = cs.next_fire(now)
        assert nf is not None
        t = time.localtime(nf)
        assert t.tm_wday == 0  # Python: Monday=0
        assert t.tm_hour == 12

    def test_next_fire_returns_future_only(self):
        cs = CronSchedule.parse("* * * * *")
        now = time.time()
        nf = cs.next_fire(now)
        assert nf > now


# ===========================================================================
# Leap year handling
# ===========================================================================

class TestLeapYearHandling:
    """Test that Feb 29 (day_of_month=29, month=2) works in leap years."""

    def test_feb_29_matches_in_leap_year(self):
        cs = CronSchedule.parse("0 12 29 2 *")
        from datetime import datetime
        ts = datetime(2024, 2, 28, 11, 59).timestamp()
        nf = cs.next_fire(ts)
        assert nf is not None
        t = time.localtime(nf)
        assert t.tm_mon == 2
        assert t.tm_mday == 29
        assert t.tm_year == 2024

    def test_feb_29_skipped_non_leap_year(self):
        cs = CronSchedule.parse("0 12 29 2 *")
        from datetime import datetime
        # 2023 was NOT a leap year. Start from late 2023 so that
        # 2024-02-29 falls within max_iter (525600 min = 365 days).
        # From 2023-02-01 to 2024-02-29 is ~394 days, which exceeds max_iter.
        ts = datetime(2023, 12, 1, 0, 0).timestamp()
        nf = cs.next_fire(ts)
        assert nf is not None
        t = time.localtime(nf)
        # Should land in 2024 (the next leap year)
        assert t.tm_year == 2024
        assert t.tm_mon == 2
        assert t.tm_mday == 29

    def test_leap_year_days_in_february(self):
        cs = CronSchedule.parse("0 12 28 2 *")
        from datetime import datetime
        # Feb 28 exists in all years; query from 2024-02-27
        ts = datetime(2024, 2, 27, 0, 0).timestamp()
        nf = cs.next_fire(ts)
        assert nf is not None
        t = time.localtime(nf)
        assert t.tm_mon == 2
        assert t.tm_mday == 28


# ===========================================================================
# ScheduledJob serialization
# ===========================================================================

class TestScheduledJob:
    """Test :class:`ScheduledJob` serialization and creation."""

    def test_create_one_shot_job(self):
        job = ScheduledJob(
            id="abc123",
            name="Reminder",
            prompt="Check the deploy",
            schedule_type=ScheduleType.ONE_SHOT,
            fire_at=time.time() + 300,
        )
        assert job.id == "abc123"
        assert job.state == JobState.PENDING
        assert job.cron is None

    def test_create_recurring_job(self):
        cs = CronSchedule.parse("0 9 * * 1-5")
        job = ScheduledJob(
            id="rec1",
            name="Daily report",
            prompt="Generate daily report",
            schedule_type=ScheduleType.RECURRING,
            cron=cs,
        )
        assert job.schedule_type == ScheduleType.RECURRING
        assert job.cron is not None

    def test_to_dict(self):
        cs = CronSchedule.parse("0 9 * * *")
        job = ScheduledJob(
            id="test1",
            name="Test job",
            prompt="Run tests",
            schedule_type=ScheduleType.RECURRING,
            cron=cs,
            fail_count=2,
            max_failures=5,
            metadata={"key": "value"},
        )
        d = job.to_dict()
        assert d["id"] == "test1"
        assert d["name"] == "Test job"
        assert d["cron"] == "0 9 * * *"
        assert d["fail_count"] == 2
        assert d["max_failures"] == 5
        assert d["metadata"]["key"] == "value"
        assert d["state"] == "PENDING"

    def test_from_dict_recurring(self):
        data = {
            "id": "test2",
            "name": "Cron job",
            "prompt": "do stuff",
            "schedule_type": "RECURRING",
            "cron": "*/10 * * * *",
            "fire_at": None,
            "state": "PENDING",
            "created_at": 1700000000.0,
            "last_fired": None,
            "last_result": None,
            "fail_count": 0,
            "max_failures": 3,
            "metadata": {},
            "agent_config": None,
        }
        job = ScheduledJob.from_dict(data)
        assert job.id == "test2"
        assert job.schedule_type == ScheduleType.RECURRING
        assert job.cron is not None
        assert job.cron.minute == "*/10"

    def test_from_dict_one_shot(self):
        data = {
            "id": "os1",
            "name": "One-shot",
            "prompt": "Do it once",
            "schedule_type": "ONE_SHOT",
            "cron": None,
            "fire_at": 1700001000.0,
            "state": "PENDING",
            "created_at": 1700000000.0,
            "last_fired": None,
            "last_result": None,
            "fail_count": 0,
            "max_failures": 3,
            "metadata": {},
            "agent_config": None,
        }
        job = ScheduledJob.from_dict(data)
        assert job.schedule_type == ScheduleType.ONE_SHOT
        assert job.cron is None
        assert job.fire_at == 1700001000.0

    def test_from_dict_missing_cron(self):
        data = {
            "id": "x",
            "name": "x",
            "prompt": "x",
            "schedule_type": "RECURRING",
            "cron": None,
        }
        job = ScheduledJob.from_dict(data)
        assert job.cron is None


# ===========================================================================
# EncreScheduler: scheduling and cancellation
# ===========================================================================

class TestEncreSchedulerBasic:
    """Test :class:`EncreScheduler` schedule and cancel methods."""

    def test_schedule_recurring(self):
        sched = EncreScheduler()
        job_id = sched.schedule(
            name="Test recurring",
            prompt="Run something",
            cron="0 9 * * *",
        )
        assert job_id is not None
        assert len(job_id) > 0
        job = sched.get_job(job_id)
        assert job is not None
        assert job.name == "Test recurring"
        assert job.schedule_type == ScheduleType.RECURRING

    def test_schedule_one_shot(self):
        sched = EncreScheduler()
        job_id = sched.schedule(
            name="Test one-shot",
            prompt="Run once",
            fire_at=time.time() + 3600,
        )
        job = sched.get_job(job_id)
        assert job is not None
        assert job.schedule_type == ScheduleType.ONE_SHOT
        assert job.fire_at is not None

    def test_schedule_no_cron_no_fire_at_defaults_immediate(self):
        sched = EncreScheduler()
        job_id = sched.schedule(name="Immediate", prompt="Go")
        job = sched.get_job(job_id)
        assert job is not None
        assert job.schedule_type == ScheduleType.ONE_SHOT
        assert job.fire_at is not None
        assert abs(job.fire_at - time.time()) < 5

    def test_cancel_existing_job(self):
        sched = EncreScheduler()
        job_id = sched.schedule(name="Cancel me", prompt="...", cron="0 9 * * *")
        assert sched.cancel(job_id) is True
        job = sched.get_job(job_id)
        assert job.state == JobState.CANCELLED

    def test_cancel_nonexistent_job(self):
        sched = EncreScheduler()
        assert sched.cancel("nonexistent") is False

    def test_cancel_all(self):
        sched = EncreScheduler()
        ids = [sched.schedule(name=f"job{i}", prompt="...", cron="0 9 * * *") for i in range(5)]
        count = sched.cancel_all()
        assert count == 5
        for jid in ids:
            assert sched.get_job(jid).state == JobState.CANCELLED

    def test_list_jobs_all(self):
        sched = EncreScheduler()
        sched.schedule(name="A", prompt="A", cron="* * * * *")
        sched.schedule(name="B", prompt="B", cron="0 0 * * *")
        jobs = sched.list_jobs()
        assert len(jobs) == 2

    def test_list_jobs_filtered(self):
        sched = EncreScheduler()
        jid = sched.schedule(name="A", prompt="A", cron="* * * * *")
        sched.cancel(jid)
        cancelled = sched.list_jobs(state=JobState.CANCELLED)
        assert len(cancelled) == 1
        # All other states are empty
        pending = sched.list_jobs(state=JobState.PENDING)
        assert len(pending) == 0

    def test_get_job_nonexistent(self):
        sched = EncreScheduler()
        assert sched.get_job("nonexistent") is None


# ===========================================================================
# EncreScheduler: durable persistence
# ===========================================================================

class TestEncreSchedulerDurability:
    """Test that durable jobs survive being written to and read from disk."""

    def test_durable_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "jobs.json")
            sched1 = EncreScheduler(durable_path=path)
            jid1 = sched1.schedule(name="Persistent", prompt="Run forever", cron="0 9 * * 1-5")
            jid2 = sched1.schedule(name="One-off", prompt="Run once", fire_at=time.time() + 99999)

            sched2 = EncreScheduler(durable_path=path)
            job1 = sched2.get_job(jid1)
            job2 = sched2.get_job(jid2)

            assert job1 is not None
            assert job1.name == "Persistent"
            assert job1.schedule_type == ScheduleType.RECURRING
            assert job1.cron is not None

            assert job2 is not None
            assert job2.name == "One-off"
            assert job2.schedule_type == ScheduleType.ONE_SHOT

    def test_durable_persists_cancel(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "jobs2.json")
            sched1 = EncreScheduler(durable_path=path)
            jid = sched1.schedule(name="Cancel me", prompt="...", cron="* * * * *")
            sched1.cancel(jid)

            sched2 = EncreScheduler(durable_path=path)
            job = sched2.get_job(jid)
            assert job is not None
            assert job.state == JobState.CANCELLED

    def test_durable_creates_parent_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "subdir", "nested", "jobs.json")
            sched = EncreScheduler(durable_path=path)
            sched.schedule(name="Nested save", prompt="...", cron="* * * * *")
            assert os.path.exists(path)

    def test_durable_no_file_no_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "nonexistent.json")
            sched = EncreScheduler(durable_path=path)
            assert sched._jobs == {}

    def test_durable_corrupted_json_recovers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "corrupt.json")
            Path(path).write_text("not valid json at all", encoding="utf-8")
            sched = EncreScheduler(durable_path=path)
            assert sched._jobs == {}

    def test_durable_bad_entry_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "partial.json")
            data = [{"id": "x"}]  # Missing required keys
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            # Should warn, not crash
            sched = EncreScheduler(durable_path=path)
            # The bad entry is skipped due to KeyError in from_dict
            assert sched._jobs == {}


# ===========================================================================
# Job lifecycle callbacks
# ===========================================================================

class TestJobCallbacks:
    def test_on_job_complete_callback_registered(self):
        sched = EncreScheduler()
        results: list[ScheduledJob] = []

        def callback(job):
            results.append(job)

        sched.on_job_complete(callback)
        assert sched._on_complete is callback

    def test_schedule_with_metadata(self):
        sched = EncreScheduler()
        jid = sched.schedule(
            name="Meta job",
            prompt="...",
            cron="0 0 * * *",
            metadata={"priority": "high", "tags": ["critical"]},
        )
        job = sched.get_job(jid)
        assert job.metadata["priority"] == "high"
        assert "critical" in job.metadata["tags"]

    def test_schedule_with_agent_config(self):
        sched = EncreScheduler()
        jid = sched.schedule(
            name="Agent job",
            prompt="...",
            cron="0 0 * * *",
            agent_config={"model": "claude-sonnet-4-20250514", "max_turns": 15},
        )
        job = sched.get_job(jid)
        assert job._agent_config is not None
        assert job._agent_config["model"] == "claude-sonnet-4-20250514"


# ===========================================================================
# Enums
# ===========================================================================

class TestEnums:
    def test_schedule_type_values(self):
        assert ScheduleType.ONE_SHOT is not None
        assert ScheduleType.RECURRING is not None
        assert ScheduleType.ONE_SHOT != ScheduleType.RECURRING

    def test_job_state_values(self):
        assert JobState.PENDING is not None
        assert JobState.RUNNING is not None
        assert JobState.COMPLETED is not None
        assert JobState.FAILED is not None
        assert JobState.CANCELLED is not None

    def test_job_state_from_string(self):
        assert JobState["PENDING"] == JobState.PENDING
        assert JobState["CANCELLED"] == JobState.CANCELLED
