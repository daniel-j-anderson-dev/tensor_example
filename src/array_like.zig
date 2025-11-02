//! Supports
//! - arrays
//! - pointer to arrays
//! - slices
//! - vectors

/// Returns the element type of the `ArrayLike` type or creates a `@compileError` if the type is not array-like
pub fn ElementType(ArrayLike: type) type {
    switch (@typeInfo(ArrayLike)) {
        .array => |array| return array.child,
        .vector => |vector| return vector.child,
        .pointer => |pointer| switch (pointer.size) {
            .slice => return pointer.child,
            .one, .many => switch (@typeInfo(pointer.child)) {
                .array => |array| return array.child,
                else => {},
            },
            else => {},
        },
        else => {},
    }
    @compileError("product is only defined for array-like types");
}

/// returns `true` if all values in `xs` are equal
pub fn allValuesEqual(xs: anytype) bool {
    for (0..xs.len - 1) |i| if (xs[i] != xs[i + 1]) return false;
    return true;
}

pub fn equal(lhs: anytype, rhs: anytype) bool {
    if (lhs.len != rhs.len) return false;
    for (lhs, rhs) |e0, e1| if (e0 != e1) return false;
    return true;
}

/// Calculates the product of all elements
pub fn product(xs: anytype) ElementType(@TypeOf(xs)) {
    var p: ElementType(@TypeOf(xs)) = 1;
    for (xs) |x| p *= x;
    return p;
}
