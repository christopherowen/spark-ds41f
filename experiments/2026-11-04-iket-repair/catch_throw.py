"""GDB-only hook: stop at exceptions thrown through the original IKET runtime."""
import gdb


class IketThrow(gdb.Breakpoint):
    def __init__(self):
        super().__init__("__cxa_throw")
        self.silent = True

    def stop(self):
        frame = gdb.newest_frame()
        found = False
        for _ in range(20):
            if frame is None:
                break
            library = gdb.solib_name(frame.pc()) or ""
            if "libsmodel_injection.so" in library:
                found = True
                break
            frame = frame.older()
        if not found:
            return False
        gdb.write("IKET_EXCEPTION_BEFORE_UNWIND\n")
        for expression in ("x/s *(char **)($x1 + 8)", "x/s *(char **)($x0 + 8)", "bt 48"):
            try:
                gdb.write(gdb.execute(expression, to_string=True))
            except gdb.error as error:
                gdb.write(str(error) + "\n")
        return True


IketThrow()
