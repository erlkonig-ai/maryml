#!/bin/bash
# Synthetic local-only regression fixtures; never invoke a real SSH command.
# Usage: bash test_gb10_lock.sh [/absolute/path/to/gb10-lock.sh]
set -uo pipefail
test_dir=$(cd -- "$(dirname -- "$0")" && pwd)
script=${1:-"$test_dir/../gb10-lock.sh"}
source_text=$(< "$script")
payload=${source_text#*<<\'REMOTE\'$'\n'}
payload=${payload%$'\nREMOTE\n'*}
fixture_root=$(mktemp -d "${GB10_LOCK_TEST_ROOT:-${TMPDIR:-/tmp}}/gb10-lock-test.XXXXXXXX")
failures=0

swap_owner() {
  /bin/mv "$case_dir/root/box.lock.d" "$case_dir/old-owner"
  /bin/mv "$case_dir/new-owner" "$case_dir/root/box.lock.d"
}
date() { printf '2000000000\n'; }
sed() {
  if [ "$fixture_case" = lookup-race ] && [ "$1" = -n ] &&
      [ "$2" = 's/^tag=//p' ] && [ -d "$case_dir/new-owner" ]; then
    /usr/bin/sed "$@"
    swap_owner
    return 0
  fi
  # Baseline refresh writes with sed; candidate writes with awk + mv.
  if [[ "$1" == -i* ]]; then
    case "$fixture_case" in
      write-failure|take-write-failure) echo 'injected write failure' >&2; return 1 ;;
      bound-race) swap_owner ;;
      installed-race) /usr/bin/sed "$@" || return; swap_owner; return ;;
    esac
  fi
  /usr/bin/sed "$@"
}
awk() {
  case "$fixture_case" in
    write-failure|take-write-failure)
      printf 'injected partial staging write\n'; return 1 ;;
    bound-race) swap_owner ;;
  esac
  /usr/bin/awk "$@"
}
mv() {
  /bin/mv "$@" || return
  [ "$fixture_case" != installed-race ] || swap_owner
}
timeout() { shift; "$@"; }
ssh() {
  # Only accept the script's exact fixture endpoint. Run its heredoc locally.
  [ "$#" = 6 ] && [ "$1" = -o ] && [ "$2" = BatchMode=yes ] &&
    [ "$3" = -o ] && [ "$4" = ConnectTimeout=10 ] &&
    [ "$5" = fixture ] || { echo 'invalid mock SSH invocation' >&2; return 99; }
  bash -c "$6"
}
export -f swap_owner date sed awk mv timeout ssh

for route in payload wrapper; do
  for root_kind in absolute relative; do
    for fixture_case in normal foreign missing-beat malformed-beat duplicate-beat write-failure take-write-failure lookup-race bound-race installed-race; do
      case_dir="$fixture_root/$route-$root_kind-$fixture_case"
      mkdir -p "$case_dir/root/box.lock.d" "$case_dir/new-owner"
      printf 'host=fixture\ntag=A\nstart=1\nbeat=1\n' > "$case_dir/root/box.lock.d/info"
      printf 'host=fixture\ntag=B\nstart=2\nbeat=2\n' > "$case_dir/new-owner/info"
      cp "$case_dir/new-owner/info" "$case_dir/replacement-info"
      case "$fixture_case" in
        foreign) cp "$case_dir/new-owner/info" "$case_dir/root/box.lock.d/info" ;;
        missing-beat) printf 'host=fixture\ntag=A\nstart=1\n' > "$case_dir/root/box.lock.d/info" ;;
        malformed-beat) printf 'host=fixture\ntag=A\nstart=1\nbeat=oops\n' > "$case_dir/root/box.lock.d/info" ;;
        duplicate-beat) printf 'beat=2\n' >> "$case_dir/root/box.lock.d/info" ;;
      esac
      cp "$case_dir/root/box.lock.d/info" "$case_dir/original-info"
      action=refresh
      [ "$fixture_case" != take-write-failure ] || action=take
      GB10_LOCK_ROOT="$case_dir/root"
      [ "$root_kind" != relative ] || GB10_LOCK_ROOT=root
      export case_dir fixture_case GB10_LOCK_ROOT
      if [ "$route" = payload ]; then
        output=$(cd "$case_dir" && bash -c "$payload" fixture "$action" A 5400 fixture "$GB10_LOCK_ROOT" 2>&1)
      else
        output=$(cd "$case_dir" && bash "$script" "$action" fixture A 2>&1)
      fi
      rc=$?
      expected=5
      case "$fixture_case" in normal) expected=0 ;; foreign|*-race) expected=3 ;; esac
      ok=yes
      [ "$rc" = "$expected" ] || ok=no
      case "$fixture_case" in
        normal)
          /usr/bin/grep -qx 'beat=2000000000' "$case_dir/root/box.lock.d/info" || ok=no
          [[ "$output" == *'BEAT A'* ]] || ok=no ;;
        *-race)
          cmp -s "$case_dir/replacement-info" "$case_dir/root/box.lock.d/info" || ok=no
          if [ "$fixture_case" = lookup-race ]; then
            cmp -s "$case_dir/original-info" "$case_dir/old-owner/info" || ok=no
          else
            /usr/bin/grep -qx 'tag=A' "$case_dir/old-owner/info" || ok=no
            /usr/bin/grep -qx 'beat=2000000000' "$case_dir/old-owner/info" || ok=no
          fi ;;
        *) cmp -s "$case_dir/original-info" "$case_dir/root/box.lock.d/info" || ok=no ;;
      esac
      if [ "$expected" != 0 ]; then
        case "$output" in *'BEAT A'*|*'TAKEN A'*|*UNREACHABLE*) ok=no ;; esac
      fi
      for tmp in "$case_dir/root/box.lock.d"/.gb10-beat.* "$case_dir/old-owner"/.gb10-beat.*; do
        [ ! -e "$tmp" ] || ok=no
      done
      printf '%s route=%s root=%s case=%s rc=%s expected=%s\n' "$ok" "$route" "$root_kind" "$fixture_case" "$rc" "$expected"
      if [ "$ok" != yes ]; then
        failures=$((failures + 1))
        printf '%s\n' "$output"
      fi
    done
  done
done
printf 'failures=%s; states=%s\n' "$failures" "$fixture_root"
[ "$failures" = 0 ]
