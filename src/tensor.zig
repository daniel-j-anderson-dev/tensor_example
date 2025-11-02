const std: type = @import("std");
const Allocator: type = std.mem.Allocator;
const ArrayList: fn (type) type = std.ArrayList;

/// Calculates the product of all elements
fn product(T: type, xs: []const T) T {
    var p: T = 1;
    for (xs) |x| p *= x;
    return p;
}

pub fn Tensor(T: type, shape_: []const usize) type {
    return struct {
        const Self = @This();
        pub const Index = [RANK]usize;

        elements: ArrayList(T), // use of the `T` parameter

        /// The rank of a `Tensor` is the number of dimensions
        pub const RANK = SHAPE.len;
        pub const SHAPE = shape_;
        pub const number_of_elements = product(usize, &SHAPE);

        pub fn rank(_: *const Self) usize {
            return RANK;
        }
        pub fn shape(_: *const Self) *const [RANK]usize {
            return SHAPE;
        }
        pub fn numberOfElements(_: *const Self) usize {
            return number_of_elements;
        }

        pub fn zeros(allocator: Allocator) Allocator.Error!Self {
            var elements = try ArrayList(T).initCapacity(allocator, Self.number_of_elements);
            for (0..number_of_elements) |_| try elements.append(allocator, 0);
            return .{ .elements = elements };
        }

        pub fn serializeIndex(index: *const Index) error{IndexOutOfBounds}!usize {
            var serialized_index: usize = 0;
            var stride: usize = 1;

            var dimension = RANK;
            while (dimension > 0) {
                dimension -= 1;
                if (index[dimension] >= SHAPE[dimension]) return error.IndexOutOfBounds;
                serialized_index += index[dimension] * stride;
                stride *= SHAPE[dimension];
            }

            return serialized_index;
        }
        pub fn deserializeIndex(index: usize) error{IndexOutOfBounds}![RANK]usize {
            var deserialized_index: Index = undefined;
            var remaining = index;

            var dimension = RANK;
            while (dimension > 0) {
                dimension -= 1;
                deserialized_index[dimension] = remaining % SHAPE[dimension];
                remaining /= SHAPE[dimension];
            }

            if (remaining != 0) return error.IndexOutOfBounds;

            return deserialized_index;
        }

        pub fn get(self: *Self, index: *const Index) error{IndexOutOfBounds}!*T {
            return &self.elements.items[try serializeIndex(index)];
        }
        pub fn set(self: *Self, index: *const Index, value: T) error{IndexOutOfBounds}!void {
            self.elements.items[try serializeIndex(index)] = value;
        }
    };
}

const test_allocator = std.testing.allocator;
const expect = std.testing.expect;

test "serialize index" {
    const ExampleTensor = Tensor(f32, &.{ 7, 1, 12 });
    var expected_serialized_index: usize = 0;
    for (0..ExampleTensor.SHAPE[0]) |i| {
        for (0..ExampleTensor.SHAPE[1]) |j| {
            for (0..ExampleTensor.SHAPE[2]) |k| {
                const actual_serialized_index = try ExampleTensor.serializeIndex(&.{ i, j, k });
                try expect(actual_serialized_index == expected_serialized_index);
                expected_serialized_index += 1;
            }
        }
    }
}

test "deserialize index" {
    const ExampleTensor = Tensor(f32, &.{ 13, 3, 9 });
    var serial_index: usize = 0;
    for (0..ExampleTensor.SHAPE[0]) |i| {
        for (0..ExampleTensor.SHAPE[1]) |j| {
            for (0..ExampleTensor.SHAPE[2]) |k| {
                const actual_deserialized_index = try ExampleTensor.deserializeIndex(serial_index);
                const expected_deserialized_index = [ExampleTensor.RANK]usize{ i, j, k };
                try expect(std.mem.eql(usize, &actual_deserialized_index, &expected_deserialized_index));
                serial_index += 1;
            }
        }
    }
}

test "product" {
    const factorial = struct {
        fn f(x: anytype) @TypeOf(x) {
            return if (x == 0) 1 else f(x - 1) * x;
        }
    }.f;
    const xs = [_]usize{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    try expect(factorial(xs.len) == product(usize, &xs));
}
